import os
# disable cuda
# os.environ['CUDA_VISIBLE_DEVICES']="-1"
import torch
import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from pathlib import Path

model_options = dict(
    device_map="cpu", 
    # load_in_8bit=True, # not with cpu
    # torch_dtype=torch.float16,
    # trust_remote_code=True
)

def main(model_repo, lora_repo = None, **download_options):
    tokenizer = AutoTokenizer.from_pretrained(model_repo, **download_options)
    model = AutoModelForCausalLM.from_pretrained(model_repo, **model_options, **download_options)
    # # FIXME move to gptq
    # if lora_repo is not None:
    #     # https://github.com/tloen/alpaca-lora/blob/main/generate.py#L40
    #     from peft import PeftModel
    #     model = PeftModel.from_pretrained(
    #         model,
    #         lora_repo, 
    #         torch_dtype=torch.float16,
    #         device_map='auto',
    #         **download_options
    #     )

def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.glob('**/*') if f.is_file())

if __name__=="__main__":
    
    # Report already downloaded models in a dataframe
    files = [dict(
        name=str(f.relative_to(HUGGINGFACE_HUB_CACHE)).replace('models--', '').replace('--', '/'),
        dir_size=sizeof_fmt(dir_size(f)),
        ctime=f.stat().st_ctime
        ) for f in Path(HUGGINGFACE_HUB_CACHE).glob('models--*')]
    df_files = pd.DataFrame(files).sort_values('ctime')
    df_files['ctime'] = pd.to_datetime(df_files['ctime'], unit='s').round('1T')
    print('models found in ', HUGGINGFACE_HUB_CACHE)
    print(df_files)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('model_repo', type=str)
    parser.add_argument('-l', '--lora_repo', type=str, default=None, help='Name of the lora repo')
    parser.add_argument('-f', '--force_download', type=str, default=None, help='Name of the lora repo')
    parser.add_argument('-r', '--resume_download', action='store_true', help='Name of the lora repo')
    args = parser.parse_args()
    
    main(args.model_repo, args.lora_repo, force_download=args.force_download, resume_download=args.resume_download, low_cpu_mem_usage=True)
