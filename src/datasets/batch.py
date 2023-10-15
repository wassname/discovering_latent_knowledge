
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from datasets.arrow_dataset import Dataset
import hashlib
import pickle
import numpy as np
from typing import List, Dict, Any, Union, NewType, Optional

from src.datasets.hs import ExtractHiddenStates
from src.helpers.typing import float_to_int16, int16_to_float
from src.helpers.ds import ds_keep_cols, clear_mem
from src.datasets.intervene import InterventionDict


def batch_hidden_states(model, tokenizer, intervention_dicts: Optional[InterventionDict], data: Dataset, batch_size=2, layer_padding=3, layer_stride=4):
    """
    Given an encoder-decoder model, a list of data, computes the contrast hidden states on n random examples.
    Returns numpy arrays of shape (n, hidden_dim) for each candidate label, along with a boolean numpy array of shape (n,)
    with the ground truth labels
    
    This is deliberately simple so that it's easy to understand, rather than being optimized for efficiency
    """
    ehs = ExtractHiddenStates(model, tokenizer, intervention_dicts=intervention_dicts, layer_stride=layer_stride, layer_padding=layer_padding)
    
    torch_cols = ['input_ids', 'attention_mask', 'choice_ids']
    ds_t_subset = ds_keep_cols(data, torch_cols)
    ds_t_subset.set_format(type='torch')
    
    ds_p_subset = data.remove_columns(torch_cols)
    
    dl = DataLoader(ds_t_subset, batch_size=batch_size, shuffle=False)
    for i, batch in enumerate(tqdm(dl, desc='get hidden states')):
        input_ids, attention_mask, choice_ids =  batch["input_ids"], batch["attention_mask"], batch["choice_ids"]
        nn = len(input_ids)
        index = i*batch_size+np.arange(nn)
        
        # different due to dropout
        hsl = ehs.get_batch_of_hidden_states(input_ids=input_ids, attention_mask=attention_mask, choice_ids=choice_ids)
        
        for j in range(nn):
            # let's add the non torch metadata like label, prompt, lie, etc
            k = i*batch_size + j
            info = ds_p_subset[k]
            
            large_arrays_keys = [k for k,v in hsl.items() if isinstance(v, torch.Tensor) and v.ndim>2]
            
            # TODO deal with multiple lists of hs in hs0
            large_arrays = {k:hsl[k][j] for k in large_arrays_keys}
            
            yield dict(
                
                # large_arrays_keys=large_arrays_keys,
                scores0=hsl["scores"][j], 
                # layer_names=hsl["layers"][j] if k==0 else [], # just in the first one, to save space
                
                ds_index=index[j],
                
                # int16 makes our storage much smaller
                **large_arrays,
                
                **info
            )
            
        info = large_arrays = hsl = None
        clear_mem()

