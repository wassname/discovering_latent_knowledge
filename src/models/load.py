"""
This file load various open source models

When editing or updating this file check out these resources:
- [LLM-As-Chatbot](https://github.com/deep-diver/LLM-As-Chatbot/blob/main/models/falcon.py)
- [oobabooga](https://github.com/oobabooga/text-generation-webui/blob/main/modules/models.py#L134)
"""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModelForCausalLM, AutoConfig, PreTrainedTokenizerBase, PreTrainedTokenizer
import torch
from src.datasets.dropout import check_for_dropout
from loguru import logger
from typing import Tuple

def verbose_change_param(tokenizer, path, after):
    before = getattr(tokenizer, path)
    if before!=after:
        setattr(tokenizer, path, after)
        logger.info(f"changing {path} from {before} to {after}")
    return tokenizer


def load_model(model_repo = "TheBloke/WizardCoder-Python-13B-V1.0-GPTQ") -> Tuple[AutoModelForCausalLM, PreTrainedTokenizerBase]:
    """
    A uncensored and large coding ones might be best for lying.
    """
    # see https://github.com/deep-diver/LLM-As-Chatbot/blob/main/models/starchat.py
    # gptq_config = GPTQConfig(bits=4, dataset="c4", disable_exllama=False)
    model_options = dict(
        device_map="auto",
        torch_dtype=torch.float16, 
    )

    config = AutoConfig.from_pretrained(model_repo, use_cache=False)
    verbose_change_param(config, 'use_cache', False)
    
    tokenizer = AutoTokenizer.from_pretrained(model_repo, use_fast=True, legacy=False)
    verbose_change_param(tokenizer, 'pad_token_id', 0)
    verbose_change_param(tokenizer, 'padding_side', 'left')
    verbose_change_param(tokenizer, 'truncation_side', 'left')
    
    model = AutoModelForCausalLM.from_pretrained(model_repo, config=config, 
                                                 **model_options)

    return model, tokenizer

