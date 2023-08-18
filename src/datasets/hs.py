from dataclasses import dataclass
import lightning as pl
import torch
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

from src.helpers.torch import to_numpy
from src.datasets.dropout import enable_dropout


from tqdm.auto import tqdm
# from src.datasets.hs import ExtractHiddenStates
from torch.utils.data import DataLoader
from datasets import Dataset
import numpy as np
import torch.nn.functional as F

default_class2choices = {False: ['No', 'Negative', 'no', 'false', 'wrong', 'False'], True: ['Yes', 'Positive', 'yes', 'true', 'correct', 'right', 'True']}

def class2choices_to_choices(class2choices):
    return [class2choices[i][0] for i in sorted(class2choices)]

def label_to_choice(label: bool, class2choices=default_class2choices) -> str:
    """turns a label like 0 to a choice like No"""
    choices = class2choices_to_choices(class2choices)
    return choices[label]

def scores2choice_probs(row, class2_ids, keys=["scores0", "scores1"] ):
    """ Given next_token scores (logits) we take only the subset the corresponds to our
    - negative tokens (e.g. False, no, ...) 
    - and positive tokens (e.g. Yes, yes, affirmative, ...).
    
    example output:
    {'choice_probs1': array([0.39, 0.31 ], dtype=float32),
    'ans1': 0.44,
    'choice_probs2': array([0.44, 0.45], dtype=float32),
    'ans2': 0.502,}
    """
    eps = 1e-5
    out = {}
    for key in keys:
        scores = row[key]
        probs = F.softmax(torch.from_numpy(scores), -1).numpy()
        probs_c = [probs[class2_ids[c]].sum() for c in class2_ids]
        
        # balance of probs
        out[key.replace("scores", "choice_probs")] = probs_c
        out[key.replace("scores", "ans")] = probs_c[1] / (np.sum(probs_c) + eps)

        # # balance of logits (much more exaggerated)
        # scores_c = [scores[class2_ids[c]].sum() for c in class2_ids]
        # out[key.replace("scores", "ansb")] = torch.tensor(scores_c).softmax(-1)[1].item()
    return out

def choice2ids(tokenizer, class2hoices: Dict[bool, List[str]]) -> Dict[int, List[int]]:
    return {k: get_choices_as_tokens(tokenizer, v) for k,v in class2hoices.items()}

def get_choices_as_tokens(
    tokenizer, choices:List[str] = ["Positive"], whitespace_first=True
) -> List[int]:
    
    # Note some tokenizers differentiate between "yes", "\nyes" and " yes", so we sometime need to add whitespace beforehand...
    if not whitespace_first:
        raise NotImplementedError('TODO')
    
    ids = []
    for c in choices:
        id_ = tokenizer(f"\n{c}", add_special_tokens=True)["input_ids"][-1]
        ids.append(id_)
        
        c2 = tokenizer.decode([id_])
        assert tokenizer.decode([id_]) == c, f'We should be able to encode and decode the choices, but it failed: tokenizer.decode(tokenizer(`{c}`))==`{c2}`!=`{c}`'

    return ids


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

        # forward pass
        last_token = -1
        with torch.no_grad():
            input_ids = input_ids.to(self.model.device)
            
            self.model.eval()
            if use_mcdropout:
                enable_dropout(self.model, use_mcdropout)

            # Forward for one step is the same as greedy generation for one step
            # https://github.com/huggingface/transformers/blob/234cfefbb083d2614a55f6093b0badfb2efc3b45/src/transformers/generation_utils.py#L1528
            model_inputs = self.model.prepare_inputs_for_generation(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            outputs = self.model.forward(
                **model_inputs,
                return_dict=True,
                output_hidden_states=True,
            )
            
            # next_token_logits = outputs.logits[:, -1, :]

            # # pre-process distribution
            # next_token_scores = logits_processor(input_ids, next_token_logits)
            # next_token_scores = logits_warper(input_ids, next_token_scores)
            # probs = nn.functional.softmax(next_token_scores, dim=-1)

            outputs["scores"] = outputs.logits[:, last_token, :]

            layers = self.get_layer_selection(outputs)
            
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

