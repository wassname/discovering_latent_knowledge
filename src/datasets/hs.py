from dataclasses import dataclass
import lightning as pl
from loguru import logger
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    AutoConfig,
    AutoModel,
    PreTrainedTokenizer,
    PreTrainedModel
)
from typing import Optional, List, Tuple, Dict
from transformers import LogitsProcessorList
import functools
from src.helpers.torch import to_numpy
from src.datasets.dropout import enable_dropout
import re

from tqdm.auto import tqdm
# from src.datasets.hs import ExtractHiddenStates
from torch.utils.data import DataLoader
from datasets import Dataset
import numpy as np
import torch
import torch.nn.functional as F
from baukit.nethook import Trace, TraceDict, recursive_copy
from einops import rearrange, reduce, repeat
from src.datasets.scores import choice2id, choice2ids
from src.helpers.torch import clear_mem


def tcopy(x: torch.Tensor):
    return x.clone().detach().cpu()

def counterfactual_loss(model, scores, token_y, token_n):
    """do a backwards pass where the loss is the distance to the opposite scores"""
    eps = 1e-4
    model.zero_grad()
    assert token_y.shape[1]<2, 'FIXME just use the first token for now'
    score_y = torch.index_select(scores, 1, token_y[:, 0])
    score_n = torch.index_select(scores, 1, token_n[:, 0])    
    # this loss would be zero if the logits of the positive and negative tokens werre flipped
    loss = F.l1_loss(score_y, score_n) + F.l1_loss(score_n, score_y)
    return loss
    

def stack_trace_returns(ret: TraceDict, names: List[str]) -> torch.Tensor:
    hs = [ret[h].output for h in names]
    return rearrange(hs, 'layers b s hs -> b layers s hs')[:, :, -1]

def stack_trace_grad_returns(ret: TraceDict, names: List[str]) -> torch.Tensor:
    hs = [ret[h].output.grad.detach() for h in names]
    return rearrange(hs, 'layers b s hs -> b layers s hs')[:, :, -1]

def select_weight_grads(weight_grads: Dict[str, torch.Tensor], pattern:str= ".+attn.c_proj.weight", mean_axis:int=1):
    grads = [g.mean(mean_axis) for k,g in weight_grads.items() if re.match(pattern, k)]
    assert len(grads), f"non of pattern='{pattern}' found in {weight_grads.keys()}"
    return rearrange(grads, "lyrs b hs -> b lyrs hs")
        
@dataclass
class ExtractHiddenStates:
    
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    layer_stride: int = 8
    layer_padding: int = 3


    def get_batch_of_hidden_states(
        self,
        input_text: Optional[List[str]] = None,
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        choice_ids: List[torch.Tensor] = None,
        truncation_length=999,
        debug=False,
    ):
        """
        Given a decoder model and a batch of texts, gets a pair of hidden states (in a given layer) on that input texts
        
        The idea is this: given two pairs of hidden states, where everything is the same except r dropout. Then tell me which one is more truthful? 
        """
        assert (input_ids is not None) or (input_text is not None), "need to provide input_ids or input_text"
        assert self.tokenizer.truncation_side == 'left'
        
        if input_text:
            raise NotImplementedError("FIXME")
            t = self.tokenizer(
                input_text,
                return_tensors="pt",
                add_special_tokens=True,
                padding='max_length', max_length=truncation_length, truncation=True, return_attention_mask=True,
            ) 
            input_ids = t.input_ids.to(self.model.device)
            attention_mask = t.attention_mask.to(self.model.device)
        else:
            input_ids = input_ids.to(self.model.device)
            attention_mask = attention_mask.to(self.model.device)
        choice_ids = choice_ids.to(self.model.device)

        # forward pass
        last_token = -1
        # for WizardLM/WizardCoder-3B-V1.0
        HEADS = [f"transformer.h.{i}.attn.c_proj" for i in range(self.model.config.num_hidden_layers)]
        MLPS = [f"transformer.h.{i}.mlp" for i in range(self.model.config.num_hidden_layers)]
        
        # for "WizardLM/WizardCoder-Python-13B-V1.0"
        HEADS = [f"model.layers.{i}.self_attn" for i in range(self.model.config.num_hidden_layers)]
        MLPS = [f"model.layers.{i}.mlp" for i in range(self.model.config.num_hidden_layers)]
        
        layers = HEADS+MLPS
        module_names = [k for k,v in self.model.named_modules()]
        layers_not_found = set(layers)-set(module_names)
        assert len(layers_not_found)==0, f"some layers not found in model: {layers_not_found}. we have {layers}"
        
        self.model.eval()
        outs = []
        with TraceDict(self.model, HEADS+MLPS, retain_grad=True, detach=True) as ret:
            with torch.autocast('cuda', torch.bfloat16):
                # Forward for one step is the same as greedy generation for one step
                # https://github.com/huggingface/transformers/blob/234cfefbb083d2614a55f6093b0badfb2efc3b45/src/transformers/generation_utils.py#L1528
                inputs_embeds = self.model.transformer.wte(input_ids)
                for _ in range(2):                
                    epsilon=inputs_embeds.abs().mean()*2 # TODO: this worked well for one prompt. Not too differen't, not to simialr. But it's a magic number
                    noise = inputs_embeds.data.new(inputs_embeds.size()).normal_(0, 1) *  epsilon
                    inputs_embeds_w_noise = inputs_embeds + noise
                    model_inputs = self.model.prepare_inputs_for_generation(input_ids=None, inputs_embeds=inputs_embeds_w_noise, attention_mask=attention_mask, use_cache=False)
                    outputs = self.model.forward(
                        **model_inputs,
                        return_dict=True,
                        output_hidden_states=True,
                    )
                    scores = outputs["scores"] = outputs.logits[:, last_token, :].float()
                    # token_n = choice_ids[:, 0] # [batch, tokens]
                    # token_y = choice_ids[:, 1]
                        
                    # loss = counterfactual_loss(self.model, scores, token_y, token_n)

                    # stack
                    hidden_states = list(outputs.hidden_states)
                    hidden_states = rearrange(hidden_states, 'lyrs b seq hs -> b lyrs seq hs')[:, :, last_token]
                    ## from ret, we get the layer activation and the grads on them
                    head_activation = tcopy(stack_trace_returns(ret, HEADS))
                    mlp_activation = tcopy(stack_trace_returns(ret, MLPS))
                    residual_stream = head_activation + mlp_activation
                    
                    # select only some layers
                    layers = self.get_layer_selection(outputs)
                    residual_stream = residual_stream[:, layers]
                    hidden_states = hidden_states[:, layers]    

                    # collect outputs
                    out = dict(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        scores=outputs["scores"],
                        layers=layers,
                        hidden_states=hidden_states,            
                        residual_stream=residual_stream,
                    )
                            
                    if debug:            
                        out['input_truncated'] = self.tokenizer.batch_decode(input_ids)
                        out['text_ans'] = self.tokenizer.batch_decode(outputs["scores"].softmax(-1).argmax(-1))
                    out = {k: detachcpu(v) for k, v in out.items()}
                    outs.append(out)
            
        # I shouldn't have to do this but I get memory leaks
        outputs = hidden_states = hidden_states2 = loss = orig_state_dict = scores = token_y = token_n = input_ids = attention_mask = choice_ids = residual_stream = residual_stream2 = None
        clear_mem()            
        return outs
    
    

    def get_layer_selection(self, outputs):
        """Sometimes we don't want to save all layers.
        
        We skip the first few (data leakage?). Stride the the middle (could be valuable), and include the last few (possibly high level concepts).

        See also https://www.lesswrong.com/posts/bWxNPMy5MhPnQTzKz/what-discovering-latent-knowledge-did-and-did-not-find-4
        """
        # for self.layer_padding, skip the first few
        strided_layers = torch.arange(
            self.layer_padding,
            len(outputs["hidden_states"])-1,
            self.layer_stride-self.layer_padding,
        )
        # for self.layer_padding ALWAYS include the last few. Why, this is based on the intuition that the last layers may be the most valuable
        last_few = torch.arange(self.layer_padding-self.layer_padding, self.layer_padding)
        layers = strided_layers+last_few
        # TODO: check for dups
        return layers

def detachcpu(x):
    """
    Trys to convert torch if possible a single item
    """
    if isinstance(x, torch.Tensor):
        # note apache parquet doesn't support half to we go for float https://github.com/huggingface/datasets/issues/4981
        x = x.detach().cpu().float()
        if x.squeeze().dim()==0:
            return x.item()
        return x
    else:
        return x
