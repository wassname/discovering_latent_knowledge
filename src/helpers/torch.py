import torch
import numpy as np
import transformers
import random
import gc

def to_numpy(x):
    """
    Trys to convert torch to numpy and if possible a single item
    """
    if isinstance(x, torch.Tensor):
        # note apache parquet doesn't support half https://github.com/huggingface/datasets/issues/4981
        x = x.detach().cpu().float()
        if x.squeeze().dim()==0:
            return x.item()
        return x.numpy()
    else:
        return x


def set_seeds(n):
    transformers.set_seed(n)
    torch.manual_seed(n)
    np.random.seed(n)
    random.seed(n)
    
def to_item(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().item()
    return x

def clear_mem():
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
