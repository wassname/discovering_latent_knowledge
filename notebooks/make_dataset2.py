# %% [markdown]
# Use pipelines as this https://github.com/wassname/representation-engineering/blob/random_comments_ignore/examples/honesty/honesty.ipynb
# 
# 

# %%
# import your package
# %load_ext autoreload
# %autoreload 2

from src.datasets.features import get_features
from src.helpers.torch import clear_mem
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.style.use('ggplot')

import os, psutil
max_dataset_memory = f"{psutil.virtual_memory().total //2}"
os.environ["HF_DATASETS_IN_MEMORY_MAX_SIZE"] = max_dataset_memory

from pathlib import Path
from tqdm.auto import tqdm
from loguru import logger
# logger.add(os.sys.stderr, format="{time} {level} {message}", level="INFO")
logger.add("logs/make_dataset_{time}.log")

from typing import Optional, List, Dict, Union, Tuple, Callable, Iterable


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch import optim
from torch.utils.data import random_split, DataLoader, TensorDataset
from simple_parsing import ArgumentParser
import transformers
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from src.repe import repe_pipeline_registry
repe_pipeline_registry()

from src.models.load import load_model
from src.extraction.config import ExtractConfig
from src.prompts.prompt_loading import load_preproc_dataset
import pickle

from src.config import root_folder
import json
# from datasets import Dataset, DatasetInfo
import datasets
from src.config import root_folder
from pathvalidate import sanitize_filename
from src.helpers.ds import ds_keep_cols

# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
# from sklearn.preprocessing import RobustScaler


# %%

TEST = False
batch_size = 2

parser = ArgumentParser(add_help=False)
parser.add_arguments(ExtractConfig, dest="run")
args = parser.parse_args()
cfg = args.run
    
# cfg = ExtractConfig(max_examples=(200, 200), model=model_name_or_path, max_length=666)
print(cfg)

model, tokenizer = load_model(cfg.model)


tokenizer_args=dict(padding="max_length", max_length=cfg.max_length, truncation=True, add_special_tokens=True)

# %%
# # cache busting for the transformers map and ds steps
# !rm -rf ~/.cache/huggingface/datasets/generator


# %% [markdown]
# # Intervention fit/load

# %%

            
def load_rep_reader(model, tokenizer, cfg, N_fit_examples=20, batch_size=2, rep_token = -1, n_difference = 1, direction_method = 'pca'):
    """
    We want one set of interventions per model
    
    So we always load a cached version if possible. to make it approx repeatable use the same dataset etc
    """
    model_name = cfg.model.replace('/', '-')
    intervention_f = root_folder / 'data' / 'interventions' / f'{model_name}.pkl'
    intervention_f.parent.mkdir(exist_ok=True, parents=True)
    if not intervention_f.exists():        
        
        hidden_layers = list(range(cfg.layer_padding, model.config.num_hidden_layers, cfg.layer_stride))
        
        dataset_fit = load_preproc_dataset('imdb', tokenizer, N=N_fit_examples, seed=cfg.seed, num_shots=cfg.num_shots, max_length=cfg.max_length)
        
        rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)
        honesty_rep_reader = rep_reading_pipeline.get_directions(
            dataset_fit['question'], 
            rep_token=rep_token, 
            hidden_layers=hidden_layers, 
            n_difference=n_difference, 
            train_labels=dataset_fit['label_true'], 
            direction_method=direction_method,
            batch_size=batch_size,
            **tokenizer_args
        )
        # and save
        with open(intervention_f, 'wb') as f:
            pickle.dump(honesty_rep_reader, f)
            logger.info(f'Saved interventions to {intervention_f}')
    else:
        with open(intervention_f, 'rb') as f:
            honesty_rep_reader = pickle.load(f)
        logger.info(f'Loaded interventions from {intervention_f}')
            
    return honesty_rep_reader


# %%

# N_fit_examples = 20
N_fit_examples = 30
rep_token = -1

honesty_rep_reader = load_rep_reader(model, tokenizer, cfg, N_fit_examples=N_fit_examples, batch_size=batch_size, rep_token=rep_token)

hidden_layers = sorted(honesty_rep_reader.directions.keys())
hidden_layers


# %% [markdown]
# # Dataset

# %%


# %% [markdown]
# # Control helpers




# %% [markdown]
# # Control

# %%


rep_control_pipeline2 = pipeline(
    "rep-control2", 
    model=model, 
    tokenizer=tokenizer, 
    layers=hidden_layers, 
    max_length=cfg.max_length,)
rep_control_pipeline2


# %%

coeff=4.0

activations = {}
for layer in hidden_layers:
    activations[layer] = torch.tensor(coeff * honesty_rep_reader.directions[layer] * honesty_rep_reader.direction_signs[layer]).to(model.device).half()
    

# %%
def metrics(control_outputs_neg, baseline_outputs, control_outputs):
    signs = [-1, 0, 1]
    for i in range(len(baseline_outputs)):
        ranked = []
        
        for j, r in enumerate([control_outputs_neg, baseline_outputs, control_outputs]):        
            choices = r[i]['answer_choices']
            label = r[i]['label_true']
            ans = r[i]['ans']
            sign = signs[j]
            ranked.append(ans)
            choice_true = choices[label]
            if label==0:
                ans *= -1            
            print(f"==== Control ({signs[j]}) ====")
            print(f"Score: {ans:02.2%} of true ans `{choice_true}`")
            # print(f"Text ans: {r['text_ans'][i]}") 
        
        is_ranked = (np.argsort(ranked)==np.arange(3)).all()
        print(f"Ranked? {is_ranked} {ranked}")
        print()



# %%
if TEST:
    # unit test: with multiple input types: single, list, generator, dataset
    ## single
    input_types = {'single':dataset_train[0], 'list':[dataset_train[i] for i in range(3)], 'generator':iter(dataset_train.select(range(3))), 'dataset':dataset_train.select(range(3)).to_iterable_dataset()}
    for name, ds in input_types.items():
        print(f"==== {name} ====")
        r = rep_control_pipeline2(ds, activations=activations, batch_size=2)
        if isinstance(r, dict):
            r = [r]
        elif isinstance(r, list):
            pass
        else:
            r = list(r)
        print(f"Control: {len(r)}")
        print(r[0]['input_ids'].shape)
        


# %%
# test intervention quality
# TODO perhaps move this to intervention create/load/cache
def test_intervention_quality(dataset_train):
    inputs = dataset_train[:3]
    activations_neg = {k:-v for k, v in activations.items()}
    activations_none = {k:v*0 for k, v in activations.items()}
    model.eval()
    with torch.no_grad():
        baseline_outputs = rep_control_pipeline2(inputs, batch_size=batch_size, activations=activations_none)
        control_outputs = rep_control_pipeline2(inputs, activations=activations, batch_size=batch_size)
        control_outputs_neg = rep_control_pipeline2(inputs, activations=activations_neg, batch_size=batch_size)


    metrics(control_outputs_neg, baseline_outputs, control_outputs)

if TEST:
    test_intervention_quality(dataset_train)

# %%


def create_hs_ds(ds_name, ds_tokens, pipeline, activations=None, f = None, batch_size=2, split_type="train", debug=TEST):
    "create a dataset of hidden states."""
    
    N = len(ds_tokens)
    dataset_name = sanitize_filename(f"{cfg.model}_{ds_name}_{split_type}_{N}", replacement_text="_")
    f = str(root_folder / '.ds'/ f"{dataset_name}")
    logger.info(f"Creating dataset {dataset_name} with {len(ds_tokens)} examples at `{f}`")
    
    info_kwargs = dict(extract_cfg=cfg.to_dict(), ds_name=ds_name, split_type=split_type, f=f, date=pd.Timestamp.now().isoformat(),)
    
    torch_cols = ['input_ids', 'attention_mask', 'choice_ids', 'question', 'answer_choices', 'example_i', 'label_true', 'sys_instr_name', 'template_name', 'instructed_to_lie']
    ds_t_subset = ds_keep_cols(ds_tokens, torch_cols)
    ds = ds_t_subset.to_iterable_dataset()
    # pipeline_it = rep_control_pipeline2(ds, batch_size=batch_size, **text_gen_kwargs)
    
    # first we make the calibration dataset with no intervention
    gen_kwargs = dict(
        model_inputs=ds,
        activations=activations,
        batch_size=batch_size,
    )
    
    if debug:
        # this allow us to debug in a single thread
        pipeline(**gen_kwargs)
    
    dataset_features = get_features(cfg, model.config)
    ds1 = datasets.Dataset.from_generator(
        generator=pipeline,
        info=datasets.DatasetInfo(
            description=json.dumps(info_kwargs, indent=2),
            config_name=f,
        ),
        gen_kwargs=gen_kwargs,
        features=dataset_features,
        num_proc=1,
        # split=split_type,
    )
    logger.info(f"Created dataset {dataset_name} with {len(ds1)} examples at `{f}`")
    ds1.save_to_disk(f)
    return ds1, f



for ds_name in cfg.datasets:
    
    # load dataset
    N=sum(cfg.max_examples)
    ds_tokens = load_preproc_dataset(ds_name, tokenizer, N=N, seed=cfg.seed, num_shots=cfg.num_shots, max_length=cfg.max_length)

    N_train_split = (len(ds_tokens) - N_fit_examples) //2
    
    N_train_split = cfg.max_examples[0]

    # split the dataset, it's preshuffled
    dataset_fit = ds_tokens.select(range(N_fit_examples))
    dataset_train = ds_tokens.select(range(N_fit_examples, N_train_split))
    dataset_test = ds_tokens.select(range(N_train_split, len(ds_tokens)))
    assert len(dataset_train)>3, f"dataset_train is too small {len(dataset_train)}"
    assert len(dataset_test)>3
    
    # FIXME:
    # test_intervention_quality(dataset_train)

    ds1, f = create_hs_ds(ds_name, dataset_train, rep_control_pipeline2, split_type="train", debug=True, batch_size=batch_size, activations=activations)
    clear_mem()
    ds1, f = create_hs_ds(ds_name, dataset_test, rep_control_pipeline2, split_type="test", debug=True, batch_size=batch_size, activations=activations)
    clear_mem()

# TODO add qc
# TODO train and test
# TODO all 3 interventions (probobly add to pipeline, and add a third dimension to the results)


# %% [markdown]
# # To Datasets
# 

# %%



