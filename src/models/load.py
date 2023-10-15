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
    Chosing:
    - https://old.reddit.com/r/LocalLLaMA/wiki/models
    - https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
    - https://github.com/deep-diver/LLM-As-Chatbot/blob/main/model_cards.json


    A uncensored and large coding ones might be best for lying.
    """
    # see https://github.com/deep-diver/LLM-As-Chatbot/blob/main/models/starchat.py
    # gptq_config = GPTQConfig(bits=4, dataset="c4", disable_exllama=False)
    model_options = dict(
        device_map="auto",
        # load_in_8bit=True,
        # load_in_4bit=True,
        # torch_dtype=torch.float16, # note because datasets pickles the model into numpy to get the unique datasets name, and because numpy doesn't support bfloat16, we need to use float16
        # use_safetensors=True,
    )

    config = AutoConfig.from_pretrained(model_repo, use_cache=False)
    verbose_change_param(config, 'use_cache', False)
    
    tokenizer = AutoTokenizer.from_pretrained(model_repo, use_fast=True)
    verbose_change_param(tokenizer, 'pad_token_id', 0)
    verbose_change_param(tokenizer, 'padding_side', 'left')
    verbose_change_param(tokenizer, 'truncation_side', 'left')
    
    model = AutoModelForCausalLM.from_pretrained(model_repo, config=config, 
                                                #  quantization_config = gptq_config, 
                                                 **model_options)

    return model, tokenizer


# def load_model(model_repo = "HuggingFaceH4/starchat-beta", lora_repo=None, verbose=True):
#     if "starchat" in model_repo:
#         model, tokenizer = load_starchat(model_repo=model_repo)
#     # elif "llama" in model_repo:
#     #     model, tokenizer = load_llama(model_repo=model_repo, lora_repo=lora_repo)
#     else:
#         raise NotImplementedError(f"code for model_repo {model_repo} not found")
    
#     if verbose: print(model.config)

#     assert check_for_dropout(model), 'model should have dropout'
#     return model, tokenizer

# def load_starchat(model_repo = "HuggingFaceH4/starchat-beta", load_in_4bit=True, torch_dtype=torch.float16):
#     # see https://github.com/deep-diver/LLM-As-Chatbot/blob/main/models/starchat.py
#     model_options = dict(
#         device_map="auto",
#         load_in_4bit=load_in_4bit,
#         torch_dtype=torch_dtype, # note because datasets pickles the model into numpy to get the unique datasets name, and because numpy doesn't support bfloat16, we need to use float16
#         use_safetensors=False,
#     )

#     config = AutoConfig.from_pretrained(model_repo, use_cache=False)
#     verbose_change_param(config, 'use_cache', False)
    
#     tokenizer = AutoTokenizer.from_pretrained(model_repo)
#     verbose_change_param(tokenizer, 'pad_token_id', 0)
#     verbose_change_param(tokenizer, 'padding_side', 'left')
#     verbose_change_param(tokenizer, 'truncation_side', 'left')
    
#     model = AutoModelForCausalLM.from_pretrained(model_repo, config=config, **model_options)

#     return model, tokenizer

# def load_llama(model_repo, lora_repo=None):
#     # https://github.com/deep-diver/LLM-As-Chatbot/blob/main/models/alpaca.py
#     model_options = dict(
#         device_map="auto", 
#         load_in_4bit=True,
#         torch_dtype=torch.float16,
#     )
        
#     tokenizer = LlamaTokenizer.from_pretrained(model_repo)
#     model = LlamaForCausalLM.from_pretrained(model_repo, **model_options)

#     if lora_repo is not None:
#         # https://github.com/tloen/alpaca-lora/blob/main/generate.py#L40
#         from peft import PeftModel
#         model = PeftModel.from_pretrained(
#             model, 
#             lora_repo, 
#             torch_dtype=torch.float16,
#             device_map='auto'
#         )
#     return model, tokenizer

# def load_falcan(model_repo, lora_repo=None):
#     # https://github.com/deep-diver/LLM-As-Chatbot/blob/main/models/falcon.py
