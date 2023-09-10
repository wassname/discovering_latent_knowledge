import gc
import torch
from datasets import Dataset

def ds_keep_cols(ds: Dataset, cols: list) -> Dataset:
    cols_all = set(ds.features.keys())
    cols_drop = cols_all-set(cols)
    return ds.remove_columns(cols_drop)

def clear_mem():
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
