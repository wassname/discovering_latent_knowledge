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
from baukit import Trace, TraceDict
from src.datasets.scores import choice2id, choice2ids

def counterfactual_backwards(model, scores, token_y, token_n):
    """do a backwards pass where the loss is the distance to the opposite scores"""
    model.zero_grad()
    assert token_y.shape[1]<2, 'FIXME just use the first token for now'
    score_y = torch.index_select(scores, 1, token_y[:, 0])
    score_n = torch.index_select(scores, 1, token_n[:, 0])
    pred = score_y - score_n
    loss = F.l1_loss(pred, -pred)
    loss.backward()

def stack_trace_returns(ret: TraceDict, HEADS: List[str]) -> torch.Tensor:
    hs = [ret[head].output.squeeze().detach().float().cpu() for head in HEADS]
    return torch.stack(hs, dim=0).squeeze().numpy()[:, -1]

@dataclass
class ExtractHiddenStates:
    
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    layer_stride: int = 1
    layer_padding: int = 2


    def get_batch_of_hidden_states(
        self,
        input_text: Optional[List[str]] = None,
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        choice_ids: List[torch.Tensor] = None,
        truncation_length=999,
        use_mcdropout=True,
        debug=False,
    ):
        """
        Given a decoder model and a batch of texts, gets a pair of hidden states (in a given layer) on that input texts
        
        The idea is this: given two pairs of hidden states, where everything is the same except r dropout. Then tell me which one is more truthful? 
        """
        assert (input_ids is not None) or (input_text is not None), "need to provide input_ids or input_text"
        assert self.tokenizer.truncation_side == 'left'
        
        if input_text:
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
        HEADS = [f"transformer.h.{i}.attn.c_proj" for i in range(self.model.config.num_hidden_layers)]
        MLPS = [f"transformer.h.{i}.mlp" for i in range(self.model.config.num_hidden_layers)]
        self.model.train()
        with TraceDict(self.model, HEADS+MLPS, retain_grad=True) as ret:
            with torch.autocast('cuda'): # FIXME not reccomended for backwards pass
                # Forward for one step is the same as greedy generation for one step
                # https://github.com/huggingface/transformers/blob/234cfefbb083d2614a55f6093b0badfb2efc3b45/src/transformers/generation_utils.py#L1528
                model_inputs = self.model.prepare_inputs_for_generation(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                outputs = self.model.forward(
                    **model_inputs,
                    return_dict=True,
                    output_hidden_states=True,
                )
                scores = outputs["scores"] = outputs.logits[:, last_token, :]
                token_n = choice_ids[:, 0] # [batch, tokens]
                token_y = choice_ids[:, 1]
            counterfactual_backwards(self.model, scores, token_y, token_n)


        # stack
        hidden_states = torch.stack(outputs.hidden_states, dim=0).squeeze()
        hidden_states = hidden_states.detach().float().cpu().numpy()[:, last_token]
        head_wise_hidden_states = stack_trace_returns(ret, HEADS)
        mlp_wise_hidden_states = stack_trace_returns(ret, MLPS)
        
        # select only some layers
        layers = self.get_layer_selection(outputs)
        head_wise_hidden_states = head_wise_hidden_states[layers]
        mlp_wise_hidden_states = mlp_wise_hidden_states[layers]
        hidden_states = hidden_states[layers]

        # collect outputs
        out = dict(
            hidden_states=hidden_states,
            scores=outputs["scores"],
            input_ids=input_ids,
            layers=layers,
            grads_attn = head_wise_hidden_states,
            grads_mlp=mlp_wise_hidden_states,
        )
        out = {k: to_numpy(v) for k, v in out.items()}
        if debug:            
            out['input_truncated'] = self.tokenizer.batch_decode(input_ids)
            out['text_ans'] = self.tokenizer.batch_decode(outputs["scores"].argmax(-1))
            
        return out

    def get_layer_selection(self, outputs):
        """Sometimes we don't want to save all layers.

        Typically we can skip some to save space (stride). We might also want to ignore the first and last ones (padding) to avoid data leakage.

        See https://www.lesswrong.com/posts/bWxNPMy5MhPnQTzKz/what-discovering-latent-knowledge-did-and-did-not-find-4
        """
        return torch.arange(
            self.layer_padding,
            len(outputs["hidden_states"]) - self.layer_padding,
            self.layer_stride,
        )

