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
from src.datasets.scores import choice2id, choice2ids


def get_gradients(model: PreTrainedModel, outputs, token_y, token_n):
    model.zero_grad()
    assert token_y.shape[1]<2, 'FIXME just use the first token for now'
    score_y = torch.index_select(outputs["scores"], 1, token_y[:, 0])
    score_n = torch.index_select(outputs["scores"], 1, token_n[:, 0])
    # score_n = outputs["scores"][:, token_n]
    pred = score_y - score_n
    loss = F.mse_loss(pred, -pred)
    loss.backward()
    ps = model.named_parameters()
    grads = {n:g.grad.cpu() for n,g in ps if g.grad is not None}
    model.zero_grad()
    # model.eval()
    return grads


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
        
        self.model.train()

        # Forward for one step is the same as greedy generation for one step
        # https://github.com/huggingface/transformers/blob/234cfefbb083d2614a55f6093b0badfb2efc3b45/src/transformers/generation_utils.py#L1528
        model_inputs = self.model.prepare_inputs_for_generation(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        outputs = self.model.forward(
            **model_inputs,
            return_dict=True,
            output_hidden_states=True,
        )

        outputs["scores"] = outputs.logits[:, last_token, :]

        layers = self.get_layer_selection(outputs)
        token_n = choice_ids[:, 0] # [batch, tokens]
        token_y = choice_ids[:, 1]
        grads_all = get_gradients(self.model, outputs, token_y, token_n)
        p = ".+mlp.c_proj.weight" # get the last weight of each layer (ignore bias)
        # p = ".+mlp.c_proj.bias" # get the last weight of each layer
        grads_mlp = torch.stack([g.mean(1).float() for k,g in grads_all.items() if re.match(p, k)])
        
        p = ".+attn.c_proj.weight" # get the last weight of each layer (ignore bias)
        grads_attn = torch.stack([g.mean(0).float() for k,g in grads_all.items() if re.match(p, k)])
        
        p = ".+mlp.c_fc.weight" # get the last weight of each layer (ignore bias)
        grads_mlp_cfc = torch.stack([g.mean(0).float() for k,g in grads_all.items() if re.match(p, k)])
        
        hidden_states = torch.stack(
            [outputs["hidden_states"][i] for i in layers], 1
        )
        # (batch, layers, past_seq, logits) take just the last token so they are same size
        hidden_states = hidden_states[
            :, :, last_token
        ]         
 
        out = dict(
            hidden_states=hidden_states,
            scores=outputs["scores"],
            input_ids=input_ids,
            layers=layers,
            grads_attn = grads_attn,
            grads_mlp=grads_mlp,
            grads_mlp_cfc=grads_mlp_cfc,
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
        return range(
            self.layer_padding,
            len(outputs["hidden_states"]) - self.layer_padding,
            self.layer_stride,
        )

