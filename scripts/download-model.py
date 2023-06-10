import os
# disable cuda
os.environ['CUDA_VISIBLE_DEVICES']="-1"
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from pathlib import Path

model_options = dict(
    device_map="auto", 
    # load_in_8bit=True, # not with cpu
    torch_dtype=torch.float16,
    trust_remote_code=True
)

def main(model_repo, lora_repo = None, **download_options):
    tokenizer = AutoTokenizer.from_pretrained(model_repo, **download_options)
    model = AutoModelForCausalLM.from_pretrained(model_repo, **model_options, **download_options)

    if lora_repo is not None:
        # https://github.com/tloen/alpaca-lora/blob/main/generate.py#L40
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model,
            lora_repo, 
            torch_dtype=torch.float16,
            device_map='auto',
            **download_options
        )




if __name__=="__main__":
    
    files = [f.relative_to(HUGGINGFACE_HUB_CACHE) for f in Path(HUGGINGFACE_HUB_CACHE).glob('models--*')]
    files = "\n".join(sorted([str(f).replace('--', '/') for f in files]))
    print(HUGGINGFACE_HUB_CACHE)
    print("Downloaded models:\n", files)
    1/0
    
    parser = argparse.ArgumentParser()
    parser.add_argument('model_repo', type=str)
    parser.add_argument('-l', '--lora_repo', type=str, default=None, help='Name of the lora repo')
    parser.add_argument('-f', '--force_download', type=str, default=None, help='Name of the lora repo')
    parser.add_argument('-r', '--resume_download', type=str, default=None, help='Name of the lora repo')
    args = parser.parse_args()
    
    main(args.model_repo, args.lora_repo, force_download=args.force_download, resume_download=args.resume_download, low_cpu_mem_usage=True)
