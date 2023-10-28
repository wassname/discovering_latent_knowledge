import numpy as np
import pandas as pd
import torch
import json
from pathlib import Path
from datasets import load_dataset, load_from_disk
# from src.helpers.typing import int16_to_float, float_to_int16


def filter_ds_to_known(ds1, verbose=True):
    """filter the dataset to only those where the model knows the answer"""
    
    # first get the rows where it answered the question correctly
    df = ds2df(ds1)
    d = df.query('sys_instr_name=="truth"').set_index("example_i")
    m1 = d.llm_ans==d.label_true
    known_indices = d[m1].index
    known_rows = df['example_i'].isin(known_indices)
    known_rows_i = df[known_rows].index
    
    if verbose: print(f"select rows are {m1.mean():2.2%} based on knowledge")
    return ds1.select(known_rows_i)


def get_ds_name(ds):
    return json.loads(ds.info.description)['ds_name']

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



def qc_ds(ds):
    
    df = ds2df(ds)
    ds_name = get_ds_name(ds)
    print('QC: ds', ds_name)
    
    # check llm accuracy
    d = df.query('instructed_to_lie==False')
    acc = (d.label_instructed==d.llm_ans).mean()
    assert np.isfinite(acc)
    print(f"\tacc    =\t{acc:2.2%} [N={len(d)}] - when the model is not lying... we get this task acc")
    assert acc>0.3, "model cannot solve task"
    
    # check LLM lie freq
    d = df.query('instructed_to_lie==True')
    acc = (d.label_instructed==d.llm_ans).mean()
    assert np.isfinite(acc)
    print(f"\tlie_acc=\t{acc:2.2%} [N={len(d)}] - when the model tries to lie... we get this acc")
    assert acc>0.01, "no known lies"
    
    # check LLM lie freq
    ds_known = filter_ds_to_known(ds, verbose=False)
    df_known = ds2df(ds_known)
    d = df_known.query('instructed_to_lie==True')
    acc = (d.label_instructed==d.llm_ans).mean()
    assert np.isfinite(acc)
    print(f"\tknown_lie_acc=\t{acc:2.2%} [N={len(d)}] - when the model tries to lie and knows the answer... we get this acc")
    assert acc>0.01, "no known lies"
    
    # check choice coverage
    mean_prob = np.sum(ds['choice_probs'], 1).mean()
    print(f"\tchoice_cov=\t{mean_prob:2.2%} - Our choices accounted for a mean probability of this")
    assert mean_prob>0.1, "neither of the available choice very unlikely :(, try debuging your templates. Check: using the correct prompt, the whitespace is correct, the correct eos_tokens (if any)"
    
    # view prompt example
    r = ds[0]
    print('prompt example:')
    print(r['input_truncated'], end="")
    print(r['text_ans'])
    
    print('='*80)
    print()
