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
from typing import Optional, List, Tuple, Dict, NewType
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
from collections import defaultdict
from dataclasses import field
from src.datasets.intervene import InterventionDict, intervention_meta_fn
from functools import partial


# def noise_for_embeds(inputs_embeds, seed=42, std = 2e-2):
#     B, S, embed_dim = inputs_embeds.shape
#     with torch.random.fork_rng(devices=[inputs_embeds.device.index]):
#         torch.manual_seed(seed)
#         noise = torch.normal(0., std, (embed_dim, ))
#         noise = repeat(noise, 't -> b s t', b=B, s=S).to(inputs_embeds.device).to(inputs_embeds.dtype)
#     return noise

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
    hs = [h[0] if isinstance(h, tuple) else h for h in hs] # from a head it's a tuple
    return rearrange(hs, 'layers b s hs -> b layers s hs')[:, :, -1]

# def stack_trace_grad_returns(ret: TraceDict, names: List[str]) -> torch.Tensor:
#     hs = [ret[h].output.grad.detach() for h in names]
#     return rearrange(hs, 'layers b s hs -> b layers s hs')[:, :, -1]

# def select_weight_grads(weight_grads: Dict[str, torch.Tensor], pattern:str= ".+attn.c_proj.weight", mean_axis:int=1):
#     grads = [g.mean(mean_axis) for k,g in weight_grads.items() if re.match(pattern, k)]
#     assert len(grads), f"non of pattern='{pattern}' found in {weight_grads.keys()}"
#     return rearrange(grads, "lyrs b hs -> b lyrs hs")
        


@dataclass
class ExtractHiddenStates:
    
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    intervention_dicts: Optional[InterventionDict] = None
    layer_stride: int = 8
    layer_padding: int = 3
    
    def get_layer_names(self):
        # for WizardLM/WizardCoder-3B-V1.0
        # HEADS = [f"transformer.h.{i}.attn.c_proj" for i in range(self.model.config.num_hidden_layers)]
        # MLPS = [f"transformer.h.{i}.mlp" for i in range(self.model.config.num_hidden_layers)]
        
        # for "WizardLM/WizardCoder-Python-13B-V1.0"
        # HACK: depends on model layout
        layers_names_h = [f"model.layers.{i}.self_attn" for i in range(self.model.config.num_hidden_layers)]
        layers_names_mlp = [f"model.layers.{i}.mlp" for i in range(self.model.config.num_hidden_layers)]
        return self.get_layer_selection(layers_names_h) + self.get_layer_selection(layers_names_mlp)


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
                
        layers_names = self.get_layer_names()
        
        self.model.eval()
        
        
        if self.intervention_dicts is not None:
            # extraction mode
            # 15 is a magic number from honest_llama
            intervention_fn1 = partial(intervention_meta_fn, interventions=self.intervention_dicts, alpha=-15)
            intervention_fn2 = partial(intervention_meta_fn, interventions=self.intervention_dicts, alpha=15)
            edit_outputs = [intervention_fn1, intervention_fn2]
        else:
            # calibration mode
            edit_outputs = [None]
        
        with torch.no_grad():
            multi_outs = defaultdict(list)
            for edit_output in edit_outputs:
                with TraceDict(self.model, layers_names, retain_grad=True, detach=True, edit_output=edit_output) as ret:
                    model_inputs = self.model.prepare_inputs_for_generation(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                    outputs = self.model.forward(
                        **model_inputs,
                        return_dict=True,
                        output_hidden_states=True,
                    )
                    outputs["scores"] = outputs.logits[:, last_token, :].float()

                    # stack
                    hidden_states = list(outputs.hidden_states)
                    hidden_states = rearrange(hidden_states, 'lyrs b seq hs -> b lyrs seq hs')[:, :, last_token]
                    ## from ret, we get the layer activation and the grads on them
                    head_activation = tcopy(stack_trace_returns(ret, layers_names))
                    # mlp_activation = tcopy(stack_trace_returns(ret, MLPS))

                    # collect outputs
                    multi_outs['scores'].append(outputs["scores"])
                    # multi_outs['hidden_states'].append(hidden_states)
                    multi_outs['head_activation'].append(head_activation)
                    # multi_outs['mlp_activation'].append(mlp_activation)
                
            # stack
            multi_outs['scores'] = torch.stack(multi_outs['scores'], -1)
            # multi_outs['mlp_activation'] = torch.stack(multi_outs['mlp_activation'], -1)
            multi_outs['head_activation'] = torch.stack(multi_outs['head_activation'], -1)
            
            # combine
            out_common = dict(input_ids=input_ids, attention_mask=attention_mask, layers=layers_names,)
            if debug:            
                out_common['input_truncated'] = self.tokenizer.batch_decode(input_ids)
                out_common['text_ans'] = self.tokenizer.batch_decode(outputs["scores"].softmax(-1).argmax(-1))
            
            out = {**multi_outs, **out_common}
            
            # detach
            out = {k: detachcpu(v) for k, v in out.items()}
            
            
        # I shouldn't have to do this but I get memory leaks
        outputs = hidden_states = hidden_states2 = loss = orig_state_dict = scores = token_y = token_n = input_ids = attention_mask = choice_ids = residual_stream = residual_stream2 = None
        clear_mem()            
        return out
    
    

    def get_layer_selection(self, layer_names):
        """Sometimes we don't want to save all layers.
        
        We skip the first few (data leakage?). Stride the the middle (could be valuable), and include the last few (possibly high level concepts).

        See also https://www.lesswrong.com/posts/bWxNPMy5MhPnQTzKz/what-discovering-latent-knowledge-did-and-did-not-find-4
        """
        module_names = [k for k,v in self.model.named_modules()]
        layers_not_found = set(layer_names)-set(module_names)
        assert len(layers_not_found)==0, f"some layers not found in model: {layers_not_found}. we have {layer_names}"
        
        # for self.layer_padding, skip the first few
        num_layers = len(layer_names)-1
        strided_layers = torch.arange(
            self.layer_padding,
            num_layers-self.layer_padding,
            self.layer_stride,
        ).tolist()
        # for self.layer_padding ALWAYS include the last few. Why, this is based on the intuition that the last layers may be the most valuable
        last_few = torch.arange(num_layers-self.layer_padding, num_layers).tolist()
        layers_inds = sorted(set(list(strided_layers)+list(last_few)))
        return [layer_names[i] for i in layers_inds]

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
