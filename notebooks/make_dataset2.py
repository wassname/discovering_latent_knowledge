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

plt.style.use("ggplot")

import os, psutil

max_dataset_memory = f"{psutil.virtual_memory().total //2}"
os.environ["HF_DATASETS_IN_MEMORY_MAX_SIZE"] = max_dataset_memory
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
from src.datasets.intervene import create_cache_interventions

# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
# from sklearn.preprocessing import RobustScaler


# %%

TEST = False

parser = ArgumentParser(add_help=False)
parser.add_arguments(ExtractConfig, dest="run")
args = parser.parse_args()
cfg = args.run
print(cfg)

batch_size = cfg.batch_size

model, tokenizer = load_model(cfg.model, pad_token_id=cfg.pad_token_id)


tokenizer_args = dict(
    padding="max_length",
    max_length=cfg.max_length,
    truncation=True,
    add_special_tokens=True,
)

# %%
if TEST:
    # # cache busting for the transformers map and ds steps
    import shutil

    shutil.rmtree("~/.cache/huggingface/datasets/generator")


# %% [markdown]
# # Intervention fit/load


# %%

# Fit an intervention

intervention = create_cache_interventions(
    model,
    tokenizer,
    cfg,
)

hidden_layers = sorted(intervention.direction.keys())
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
    max_length=cfg.max_length,
)
rep_control_pipeline2


# %%
# from src.datasets.intervene import get_activations_from_reader
from src.datasets.intervene import test_intervention_quality


# %%
# test intervention quality
# TODO perhaps move this to intervention create/load/cache

# %%


def create_hs_ds(
    ds_name,
    ds_tokens,
    pipeline,
    intervention=None,
    f=None,
    batch_size=2,
    split_type="train",
    debug=TEST,
):
    "create a dataset of hidden states." ""

    N = len(ds_tokens)
    dataset_name = sanitize_filename(
        f"{cfg.model}_{ds_name}_{split_type}_{N}", replacement_text="_"
    )
    f = str(root_folder / ".ds" / f"{dataset_name}")
    logger.info(
        f"Creating dataset {dataset_name} with {len(ds_tokens)} examples at `{f}`"
    )

    info_kwargs = dict(
        extract_cfg=cfg.to_dict(),
        ds_name=ds_name,
        split_type=split_type,
        f=f,
        date=pd.Timestamp.now().isoformat(),
    )

    torch_cols = [
        "input_ids",
        "attention_mask",
        "choice_ids",
        "question",
        "answer_choices",
        "example_i",
        "label_true",
        "sys_instr_name",
        "template_name",
        "instructed_to_lie",
    ]
    ds_t_subset = ds_keep_cols(ds_tokens, torch_cols)
    ds = ds_t_subset.to_iterable_dataset()
    # pipeline_it = rep_control_pipeline2(ds, batch_size=batch_size, **text_gen_kwargs)

    # first we make the calibration dataset with no intervention
    gen_kwargs = dict(
        model_inputs=ds,
        intervention=intervention,
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


if cfg.disable_ds_cache:
    from datasets import disable_caching

    disable_caching()

from src.datasets.load import ds2df, load_ds, get_ds_name, filter_ds_to_known, qc_ds

for ds_name in cfg.datasets:
    N_train, N_test = cfg.max_examples
    # load dataset
    N = sum(cfg.max_examples)
    ds_tokens = load_preproc_dataset(
        ds_name,
        tokenizer,
        N=N,
        seed=cfg.seed,
        num_shots=cfg.num_shots,
        max_length=cfg.max_length,
        prompt_format=cfg.prompt_format,
    )

    assert len(ds_tokens) >= N, f"dataset is too small as {len(ds_tokens)}< {N}"

    # N_train_split = (len(ds_tokens) - cfg.intervention_fit_examples) // 2
    assert cfg.intervention_fit_examples < N_train
    N_train_split = N_train - cfg.intervention_fit_examples

    # split the dataset, it's preshuffled
    dataset_fit = ds_tokens.select(range(cfg.intervention_fit_examples))
    dataset_train = ds_tokens.select(
        range(cfg.intervention_fit_examples, N_train_split)
    )
    dataset_test = ds_tokens.select(range(N_train_split, len(ds_tokens)))
    assert len(dataset_train) > 3, f"dataset_train is too small {len(dataset_train)}"
    assert len(dataset_test) > 3

    # FIXME:
    test_intervention_quality(
        dataset_train,
        intervention,
        model,
        rep_control_pipeline2,
        batch_size=batch_size,
        ds_name=ds_name,
    )

    ds1, f = create_hs_ds(
        ds_name,
        dataset_train,
        rep_control_pipeline2,
        split_type="train",
        debug=True,
        batch_size=batch_size,
        intervention=intervention,
    )
    clear_mem()
    ds1, f = create_hs_ds(
        ds_name,
        dataset_test,
        rep_control_pipeline2,
        split_type="test",
        debug=True,
        batch_size=batch_size,
        intervention=intervention,
    )
    clear_mem()

    try:
        qc_ds(ds1)
    except:
        raise
        logger.exception(f"QC failed for {ds_name}")

# TODO add qc
# TODO train and test
# TODO all 3 interventions (probobly add to pipeline, and add a third dimension to the results)


# %% [markdown]
# # To Datasets
#

# %%
