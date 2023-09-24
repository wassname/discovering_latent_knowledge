# # Lets save our data as a huggingface dataset, so it's quick to reuse


import psutil, os
max_dataset_memory = f"{psutil.virtual_memory().total //2}"
os.environ["HF_DATASETS_IN_MEMORY_MAX_SIZE"] = max_dataset_memory

from datasets import disable_caching
disable_caching()

from loguru import logger
logger.add("make_dataset_{time}.log")

import pandas as pd
import numpy as np

from typing import Optional, List, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import pickle
import hashlib
from pathlib import Path

import transformers
from datasets import Dataset, DatasetInfo
from src.datasets.load import load_ds
from src.models.load import verbose_change_param, AutoConfig, AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
import os, re, sys, collections, functools, itertools, json

from src.models.load import load_model
from src.datasets.load import ds2df
from src.datasets.load import rows_item
from src.datasets.batch import batch_hidden_states
# from src.datasets.scores import choice2ids, scores2choice_probs


from itertools import chain
import functools
from src.prompts.prompt_loading import load_prompts
from src.datasets.scores import scores2choice_probs
from src.datasets.scores import choice2id, choice2ids

from simple_parsing import ArgumentParser
from src.extraction.config import ExtractConfig
parser = ArgumentParser(add_help=False)
parser.add_arguments(ExtractConfig, dest="run")

# argv="""\
# "WizardLM/WizardCoder-Python-13B-V1.0" \
# imdb amazon_polarity super_glue:boolq glue:qnli \
# --max_examples 260 260 \
# --max_length=600 \
# --num_shots=1 \
# """.strip().replace('\n','').split()
# print(argv)

# argv="""\
# "HuggingFaceH4/starchat-beta" \
# imdb amazon_polarity super_glue:boolq glue:qnli \
# --max_examples 260 260 \
# --max_length=600 \
# --num_shots=1 \
# """.strip().replace('\n','').split()
# print(argv)

args = parser.parse_args()
cfg = args.run
print(cfg)

BATCH_SIZE = 1  # None # None means auto # 6 gives 16Gb/25GB. where 10GB is the base model. so 6 is 6/15



def load_model(model_repo = "HuggingFaceH4/starchat-beta"):
    """
    Chosing:
    - https://old.reddit.com/r/LocalLLaMA/wiki/models
    - https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
    - https://github.com/deep-diver/LLM-As-Chatbot/blob/main/model_cards.json


    A uncensored and large coding ones might be best for lying.
    """
    # see https://github.com/deep-diver/LLM-As-Chatbot/blob/main/models/starchat.py
    model_options = dict(
        device_map="auto",
        # load_in_8bit=True,
        load_in_4bit=True,
        torch_dtype=torch.float16, # note because datasets pickles the model into numpy to get the unique datasets name, and because numpy doesn't support bfloat16, we need to use float16
        # use_safetensors=False,
    )

    config = AutoConfig.from_pretrained(model_repo, use_cache=False)
    verbose_change_param(config, 'use_cache', False)
    
    tokenizer = AutoTokenizer.from_pretrained(model_repo)
    verbose_change_param(tokenizer, 'pad_token_id', 0)
    verbose_change_param(tokenizer, 'padding_side', 'left')
    verbose_change_param(tokenizer, 'truncation_side', 'left')
    
    model = AutoModelForCausalLM.from_pretrained(model_repo, config=config, **model_options)

    return model, tokenizer


def qc_ds(f):
    ds4 = load_ds(f)

    # QC: check that the dataset is valid
    for k,v in ds4[0].items():
        print(k, v.shape, v.dtype)
        if (isinstance(v, (np.ndarray, np.generic, torch.Tensor)) and (v.dtype in ['float16', 'float32', 'float64', 'int64', 'int32', 'int16', 'int8'])):
            assert np.isfinite(v).all()


    # QC, check which answers are most common
    common_answers = pd.Series(ds4['txt_ans0']).value_counts()
    print('Remember it should be binary. Found common LLM answers:', common_answers)

    current_choices = set(list(chain(*ds4['answer_choices'])))
    unexpected_answers = set(common_answers.head(10).index)-current_choices
    if len(unexpected_answers):
        logger.warning(f'found unexpected answers: {unexpected_answers}. You may want to add them to class2choices')
    
    mean_prob = ds4['choice_probs0'].sum(-1).mean()
    print('mean_prob', mean_prob)
    assert ds4['choice_probs0'].sum(-1).mean()>0.2, f"""
    Our choices should cover most common answers. But they accounted for a mean probability of {mean_prob:2.2%} (should be >40%). 

    To fix this you might want to improve your prompt or add to your choices
    """

    df = ds2df(ds4)
    print(df.head(5))

    # QC check accuracy
    # it should manage to lie some of the time when asked to lie. Many models wont lie unless very explicitly asked to, but we don't want to do that, we want to leave some ambiguity in the prompt

    d = df.query('instructed_to_lie==True')
    acc = (d.label_instructed==d.llm_ans).mean()
    print(f"when the model tries to lie... we get this acc {acc:2.2f}")
    assert acc>0.1, f"should be acc>0.1 but is acc={acc}"

    # ### QC stats
    def stats(df):
        return dict(
            acc=(df.llm_ans == df.label_instructed).mean(),
            n=len(df),
        )
        
    def col2statsdf(df, group):
        return pd.DataFrame(df.groupby(group).apply(stats).to_dict()).T
        
        
    print("how well does it do the simple task of telling the truth, for each template")
    col2statsdf(df.query('sys_instr_name=="truth"'), 'template_name')

    print("how well does it complete the task for each prompt")
    # of course getting it to tell the truth is easy, but how effective are the other prompts?
    col2statsdf(df, 'sys_instr_name')

    # ### QC view row

    # QC by viewing a row
    r = ds4[0]
    print(r['prompt_truncated'])
    print(r['txt_ans0'])

    # # QC: generation
    # 
    # Let's a quick generation, so we can QC the output and sanity check that the model can actually do the task

    # r = ds[2]
    # q = r["prompt_truncated"]

    # pipeline = transformers.pipeline(
    #     "text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    # )
    # sequences = pipeline(
    #     q.lstrip('<|endoftext|>'),
    ##     max_length=100,
    # max_new_tokens=10,
    #     do_sample=False,
    #     return_full_text=False,
    #     eos_token_id=tokenizer.eos_token_id,
    # )

    # for seq in sequences:
    #     print("-" * 80)
    #     print(q)
    #     print("-" * 80)
    #     print(f"`{seq['generated_text']}`")
    #     print("-" * 80)
    #     print("label", r['label'])


    # # QC: linear probe

    from sklearn.preprocessing import RobustScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

    # # just select the question where the model knows the answer. 
    df = ds2df(ds4)
    d = df.query('sys_instr_name=="truth"').set_index("example_i")

    # # these are the ones where it got it right when asked to tell the truth
    m1 = d.llm_ans==d.label_true
    known_indices = d[m1].index
    print(f"select rows are {m1.mean():2.2%} based on knowledge")
    # # convert to row numbers, and use datasets to select
    known_rows = df['example_i'].isin(known_indices)
    known_rows_i = df[known_rows].index

    # # also restrict it to significant permutations. That is monte carlo dropout pairs, where the answer changes by more than X%
    # m = np.abs(df.ans0-df.ans1)>0.05
    # print(f"selected rows are {m.mean():2.2%} for significance")
    # significant_rows = m[m].index

    # allowed_rows_i = set(known_rows_i).intersection(significant_rows)
    # allowed_rows_i = significant_rows
    ds5 = ds4.select(known_rows_i)
    df = ds2df(ds5)


    large_arrays_keys = [k for k,v in ds4[0].items() if v.ndim>1]

    for k in large_arrays_keys:
        print('-'*80)
        print(k)
        hs = ds5[k]
        X = hs.reshape(hs.shape[0], -1)


        y = df['label_true'] == df['llm_ans']

        # split
        n = len(y)
        max_rows = 1000
        
        X_train, X_test = X[:n//2], X[n//2:]
        y_train, y_test = y[:n//2], y[n//2:]
        X_train = X_train[:max_rows]
        y_train = y_train[:max_rows]
        X_test = X_test[:max_rows]
        y_test = y_test[:max_rows]
        print('split size', X_train.shape, y_test.shape)

        # scale
        scaler = RobustScaler()
        scaler.fit(X_train)
        X_train2 = scaler.transform(X_train)
        X_test2 = scaler.transform(X_test)

        lr = LogisticRegression(class_weight="balanced", penalty="l2", max_iter=380)
        lr.fit(X_train2, y_train>0)

        print("Logistic cls acc: {: 3.2%} [TRAIN]".format(lr.score(X_train2, y_train>0)))
        print("Logistic cls acc: {: 3.2%} [TEST]".format(lr.score(X_test2, y_test>0)))


    # QC: make sure we didn't lose all of the successful lies, which would make the problem trivial
    df2= ds2df(ds5)
    df_subset_successull_lies = df2.query("instructed_to_lie==True & (llm_ans==label_instructed)")
    print(f"filtered to {len(df_subset_successull_lies)} num successful lies out of {len(df2)} dataset rows")
    assert len(df_subset_successull_lies)>0, "there should be successful lies in the dataset"

    print(f)



     
# TODO: loop through all prompts in this dataset
ds_names = cfg.datasets
split_type = "train"

model, tokenizer = load_model(cfg.model)

def row_choice_ids(r):
    return choice2ids([[c] for c in r['answer_choices']], tokenizer)


for ds_name in ds_names:
        
    # TODO: for when we need custom templates....
    # ds_root_name, _, subset_name = ds_name.partition(":")
    # template_path = cfg.template_path/ds_root_name
    # if subset_name:
    #     template_path = template_path/subset_name
    # template_path

    # NOTE: you may need to `rm ~/.cache/huggingface/datasets/generator`
    N = cfg.max_examples[split_type!="train"]
    ds_prompts = Dataset.from_generator(
        load_prompts, 
        gen_kwargs=dict(
            ds_string=ds_name, 
            num_shots=cfg.num_shots,
            split_type=split_type,
            # template_path=template_path,
            seed=cfg.seed,
            prompt_format='llama',
            N=N*3,
        ), 
        )

    # ## Format prompts
    # The prompt is the thing we most often have to change and debug. So we do it explicitly here.
    # We do it as transforms on a huggingface dataset.
    # In this case we use multishot examples from train, and use the test set to generated the hidden states dataset. We will test generalisation on a whole new dataset.

    ds_tokens = (
        ds_prompts
        .map(
            lambda ex: tokenizer(
                ex["question"], padding="max_length", max_length=cfg.max_length, truncation=True, add_special_tokens=True,
                return_tensors="np",
                return_attention_mask=True,
                # return_overflowing_tokens=True,
            ),
            batched=True,
        )
        .map(lambda r: {"truncated": np.sum(r["attention_mask"], -1)==cfg.max_length})
        .map(
            lambda r: {"prompt_truncated": tokenizer.batch_decode(r["input_ids"])},
            batched=True,
        )
        .map(lambda r: {'choice_ids': row_choice_ids(r)})
    )
    
    
    ds_tokens = ds_tokens.filter(lambda r: r['truncated']==False)
    ds_tokens = ds_tokens.select(range(min(len(ds_tokens), N)))
    print('removed truncated rows to leave: num_rows', ds_tokens.num_rows)

    # ## Save as Huggingface Dataset
    # get dataset filename
    sanitize = lambda s:s.replace('/', '').replace('-', '_') if s is not None else s
    dataset_name = f"{sanitize(cfg.model)}_{ds_name}_{split_type}_{N}"
    f = f"../.ds/{dataset_name}"

    gen_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        data=ds_tokens,
        batch_size=BATCH_SIZE,
        layer_padding=cfg.layer_padding,
        layer_stride=cfg.layer_stride,
    )

    info_kwargs = dict(extract_cfg=cfg.to_dict(), ds_name=ds_name, split_type=split_type, f=f, date=pd.Timestamp.now().isoformat(),)

    # [DatasetInfo](https://github.com/huggingface/datasets/blob/9b21e181b642bd55b3ef68c1948bfbcd388136d6/src/datasets/info.py#L94)
    ds1 = Dataset.from_generator(
        generator=batch_hidden_states,
        info=DatasetInfo(
            description=json.dumps(info_kwargs, indent=2),
            config_name=f,
        ),
        gen_kwargs=gen_kwargs,
        num_proc=1,
    )

    # ## Add labels
    # For our probe. Given next_token scores (logits) we take only the subset the corresponds to our negative tokens (e.g. False, no, ...) and positive tokens (e.g. Yes, yes, affirmative, ...).
    def expand_choices(choices: List[str]) -> List[str]:
        """expand out choices by adding versions that are upper, lower, whitespace, etc"""
        new = []
        for c in choices:
            new.append(c)
            new.append(c.upper())
            new.append(c.capitalize())
            new.append(c.lower())
        return set(new)


    left_choices = list(r[0] for r in ds1['answer_choices'])+['no', 'false', 'negative', 'wrong']
    right_choices = list(r[1] for r in ds1['answer_choices'])+['yes', 'true', 'positive', 'right']
    left_choices, right_choices = expand_choices(left_choices), expand_choices(right_choices)
    expanded_choices = [left_choices, right_choices]
    expanded_choice_ids = choice2ids(expanded_choices, tokenizer)

    # this is just based on pairs for that answer...
    add_txt_ans0 = lambda r: {'txt_ans0': tokenizer.decode(r['scores0'].argmax(-1))}

    # Either just use the template choices
    add_ans = lambda r: scores2choice_probs(r, row_choice_ids(r), keys=["scores0"])

    # Or all expanded choices
    add_ans_exp = lambda r: scores2choice_probs(r, expanded_choice_ids, prefix="expanded_")
    ds1.set_format(type='numpy')#, columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
    ds3 = (
        ds1
        .map(add_ans)
        .map(add_ans_exp)
        .map(add_txt_ans0)
    )

    ds3.save_to_disk(f)
    print('! saved f=', f)
    
    try:
        qc_ds(f)
    except Exception as e:
        print('QC failed', e)
        # raise e

