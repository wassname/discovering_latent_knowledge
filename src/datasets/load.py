import numpy as np
import pandas as pd
from datasets import load_dataset
from src.helpers.typing import int16_to_float, float_to_int16

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
    if cols is None:
        r = ds[0]
        # get all the columns that not large lists or arrays
        cols = [k for k,v in r.items() if (isinstance(v, np.ndarray) and v.size<3) or not isinstance(v, (list, np.ndarray))]
    
    df = ds.select_columns(cols)
    df = pd.DataFrame([rows_item(r) for r in df])
    
    # derived
    df['dir_true'] = df['ans0']
    # df['conf'] = (df['ans0']).abs9)
    # df['llm_prob'] = (df['ans0']+df['ans1'])/2
    df['llm_ans'] = df['ans0']>0.5
    # df['desired_ans'] = df.label ^ df.lie
    return df

def load_ds(f):
    ds = load_dataset(f)
    return ds.map(lambda x: {'hs0': int16_to_float(x['hs0']), 'hs1': int16_to_float(x['hs1'])})
