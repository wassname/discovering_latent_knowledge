
from typing import Optional, List, Tuple, Dict
import numpy as np
import torch
import torch.nn.functional as F
import functools

from transformers import (
    PreTrainedTokenizer,
    PreTrainedModel
)

default_class2choices = {False: ['No', 'Negative', 'no', 'false', 'wrong', 'False'], True: ['Yes', 'Positive', 'yes', 'true', 'correct', 'right', 'True']}

# def class2choices_to_choices(class2choices):
#     return [class2choices[i][0] for i in sorted(class2choices)]

# def label_to_choice(label: bool, class2choices=default_class2choices) -> str:
#     """turns a label like 0 to a choice like No"""
#     choices = class2choices_to_choices(class2choices)
#     return choices[label]

def scores2choice_probs(row, class2_ids: List[List[int]], keys=["scores0", "scores1"], prefix=""):
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
        probs_c = [sum([probs[cc] for cc in c]) for c in class2_ids]
        
        # balance of probs
        out[prefix+key.replace("scores", "choice_probs")] = probs_c
        out[prefix+key.replace("scores", "ans")] = probs_c[1] / (np.sum(probs_c) + eps)

        # # balance of logits (much more exaggerated)
        # scores_c = [scores[class2_ids[c]].sum() for c in class2_ids]
        # out[key.replace("scores", "ansb")] = torch.tensor(scores_c).softmax(-1)[1].item()
    return out

@functools.lru_cache()
def choice2id(tokenizer, c: str, whitespace_first=True) -> int:
    """convert a choice to a single token"""
    
    # Note some tokenizers differentiate between "yes", "\nyes" and " yes", so we sometime need to add whitespace beforehand...
    prefix= "\n" if whitespace_first else ""
    text = f"{prefix}{c}"
    input_ids = tokenizer(text, add_special_tokens=True)["input_ids"]
    # assert len(input_ids)==int(bool(prefix))+1, f"`{text} should decode into 1 token and possible prefix"
    id_ = input_ids[1]
    
    # check that we can decode it
    c2 = tokenizer.decode([id_])
    # assert tokenizer.decode([id_]) == c, f'We should be able to encode and decode the choices, but it failed: tokenizer.decode(tokenizer(`{c}`))==`{c2}`!=`{c}`'
    return id_

def choice2ids(all_choices: List[List[str]], tokenizer: PreTrainedTokenizer) -> List[List[int]]:
    return [[choice2id(tokenizer, c) for c in choices] for choices in all_choices]

# def get_choice_as_token(tokenizer, choice: str) -> int:
#     return get_choices_as_tokens(tokenizer, [choice])[0]

# def get_choices_as_tokens(
#     tokenizer, choices:List[str] = ["Positive"], whitespace_first=True
# ) -> List[int]:
    
    
#     ids = []
#     for c in choices:
#         try:
#             id_ = choice2id(tokenizer, c)
#             ids.append(id_)
#         except AssertionError as e:
#             print(e)
#     return ids
