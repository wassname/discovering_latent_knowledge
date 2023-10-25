import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datasets import load_dataset, load_from_disk
# from src.helpers.typing import int16_to_float, float_to_int16

def rows_item(row):
    """
    transform a row by turning singe dim arrays into items
    """
    for k,x in row.items():
        if isinstance(x, np.ndarray) and (x.ndim==0 or (x.ndim==1 and len(x)==1)):
            row[k]=x.item()
        if isinstance(x, list) and len(x)==1:
            row[k]=x[0]
    return row


def ds2df(ds, cols=None):
    """one of our custom datasets into a dataframe
    
    dropping the large arrays and lists"""
    
    # json.loads(dss[0].info.description)['f'] # doesn't work when concat

    if cols is None:
        r = ds[0]
        # get all the columns that not large lists or arrays
        cols = [k for k,v in r.items() if (isinstance(v, np.ndarray) and v.size<3) or not isinstance(v, (list, np.ndarray))]
    ds = ds.with_format('numpy')
    df = ds.select_columns(cols)
    df = pd.DataFrame([rows_item(r) for r in df])
    
    # derived
    df['ans'] = ds['ans'].mean(-1)
    # df['dir_true'] = df['ans0'][:, 0]
    df['conf'] = (df['ans']).abs()
    df['llm_prob'] = ds['ans'].mean(-1)
    df['llm_ans'] = df['ans']>0.5
    df['label_instructed'] = df['label_true'] ^ df['instructed_to_lie']
    # df['desired_ans'] = df.label ^ df.lie
    return df

def load_ds(f):
    ds = load_from_disk(f)
    ds = ds.with_format('numpy')
    # ds['ds_name'] = Path(f).stem
    # ks = [k for k,v in ds[0].items() if (isinstance(v, (np.ndarray, np.generic, torch.Tensor) )) and (v.dtype=='int64') and k not in ['ds_index']]
    # ds = ds.map(lambda x: {k: int16_to_float(torch.from_numpy(ds[k]).long()) for k in ks})
    return ds
