
from typing import Optional, List, Tuple, Dict
import numpy as np
import torch
import torch.nn.functional as F
import functools
import itertools
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

def scores2choice_probs(row, class2_ids: List[List[int]], keys=["scores0"], prefix=""):
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
        scores = torch.tensor(row[key])
        probs = F.softmax(scores, 0).numpy() # shape [tokens, inferences)
        probs_c = [np.sum([probs[cc] for cc in c], 0) for c in class2_ids] # sum over alternate choices e.g. [['decrease', 'dec'],['inc', 'increase']]
        probs_c = np.stack(probs_c, 0) # shape [choices, inferences]
        
        # balance of probs
        out[prefix+key.replace("scores", "choice_probs")] = probs_c
        out[prefix+key.replace("scores", "ans")] = probs_c[1] / (np.sum(probs_c, 0) + eps) # shape is [inferences]

        # # balance of logits (much more exaggerated)
        # scores_c = [scores[class2_ids[c]].sum() for c in class2_ids]
        # out[key.replace("scores", "ansb")] = torch.tensor(scores_c).softmax(-1)[1].item()
    return out

# @functools.lru_cache()
def choice2id(tokenizer, c: str, whitespace_first=False) -> List[int]:
    """convert a choice to a single token"""
    # HACK: this whole function is messy, and specific to the llama tokenizer :(. I don't want it to fail silently, so I'm adding a few asserts. It's better to find out before 4 hours of data collection
    
    # Note some tokenizers differentiate between "yes", "\nyes" and " yes", and ideally we want all! 
    ids = [
        tokenizer(f' {c}', add_special_tokens=False)["input_ids"][1],
        tokenizer(f'\n{c}', add_special_tokens=False)["input_ids"][2],
        tokenizer(f'{c}', add_special_tokens=False)["input_ids"][0],
    ]
    ids = list(set(ids))
    
    # QC: they should all decode to the same token
    decoded_ids = tokenizer.batch_decode(ids)
    shortest = sorted(decoded_ids, key=lambda s:len(s))[0]
    assert all([decoded_ids[i].startswith(shortest) for i in range(len(decoded_ids))]), f"decoded_ids={decoded_ids}"    
    
    # check that we can decode it
    c3 = tokenizer.batch_decode(ids)
    for c2 in c3:
        if not c.startswith(c2) and len(c2):
            print(c, c2, c3)
            ids = tokenizer(c, add_special_tokens=False)["input_ids"]
            decoded_ids = [tokenizer.decode(i) for i in ids]
            print(f"{c}=>{ids}=>{decoded_ids}")
            raise AssertionError(f'We should be able to encode and decode the choices, but it failed: tokenizer.decode(tokenizer(`{c}`))==`{c2}`!=`{c}`')
    return ids

def choice2ids(all_choices: List[List[str]], tokenizer: PreTrainedTokenizer) -> List[List[int]]:
    choices = [list(itertools.chain(*[choice2id(tokenizer, c) for c in choices])) for choices in all_choices]
    assert choices[0]!=choices[1], "choices should be different"
    assert choices[0][0]!=choices[1][0], "choices should be different"
    return choices

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
