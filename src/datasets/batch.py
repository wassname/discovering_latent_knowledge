
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from datasets.arrow_dataset import Dataset
import hashlib
import pickle
import numpy as np

from src.datasets.hs import ExtractHiddenStates
from src.helpers.typing import float_to_int16, int16_to_float
from src.helpers.ds import ds_keep_cols, clear_mem


def batch_hidden_states(model, tokenizer, data: Dataset, batch_size=2, layer_padding=3, layer_stride=4):
    """
    Given an encoder-decoder model, a list of data, computes the contrast hidden states on n random examples.
    Returns numpy arrays of shape (n, hidden_dim) for each candidate label, along with a boolean numpy array of shape (n,)
    with the ground truth labels
    
    This is deliberately simple so that it's easy to understand, rather than being optimized for efficiency
    """
    ehs = ExtractHiddenStates(model, tokenizer, layer_stride=layer_stride, layer_padding=layer_padding)
    
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
        hs0 = ehs.get_batch_of_hidden_states(input_ids=input_ids, attention_mask=attention_mask, choice_ids=choice_ids)
        
        for j in range(nn):
            # let's add the non torch metadata like label, prompt, lie, etc
            k = i*batch_size + j
            info = ds_p_subset[k]
            
            large_arrays_keys = [k for k,v in hs0.items() if v.ndim>2]
            large_arrays_as_int16 = {
                # k:float_to_int16(hs0[k][j])
                k:hs0[k][j] 
                for k in large_arrays_keys}
            
            yield dict(
                
                # large_arrays_keys=large_arrays_keys,
                scores0=hs0["scores"][j],            
                
                ds_index=index[j],
                
                # int16 makes our storage much smaller
                **large_arrays_as_int16,
                
                **info
            )
            
        info = large_arrays_as_int16= hs0 = None
        clear_mem()

