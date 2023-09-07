
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from datasets.arrow_dataset import Dataset
import hashlib
import pickle
import numpy as np

from src.datasets.hs import ExtractHiddenStates
from src.helpers.typing import float_to_int16, int16_to_float
from src.helpers.ds import ds_keep_cols


def batch_hidden_states(model, tokenizer, data: Dataset, batch_size=2, mcdropout=True):
    """
    Given an encoder-decoder model, a list of data, computes the contrast hidden states on n random examples.
    Returns numpy arrays of shape (n, hidden_dim) for each candidate label, along with a boolean numpy array of shape (n,)
    with the ground truth labels
    
    This is deliberately simple so that it's easy to understand, rather than being optimized for efficiency
    """
    ehs = ExtractHiddenStates(model, tokenizer)
    
    torch_cols = ['input_ids', 'attention_mask', 'choice_ids']
    ds_t_subset = ds_keep_cols(data, torch_cols)
    ds_t_subset.set_format(type='torch')
    
    ds_p_subset = data.remove_columns(torch_cols)
    # TODO check it has a few critical ones in
    
    dl = DataLoader(ds_t_subset, batch_size=batch_size, shuffle=False)
    for i, batch in enumerate(tqdm(dl, desc='get hidden states')):
        input_ids, attention_mask, choice_ids =  batch["input_ids"], batch["attention_mask"], batch["choice_ids"]
        nn = len(input_ids)
        index = i*batch_size+np.arange(nn)
        
        # different due to dropout
        hs0 = ehs.get_batch_of_hidden_states(input_ids=input_ids, attention_mask=attention_mask, use_mcdropout=mcdropout, choice_ids=choice_ids)

        
        for j in range(nn):
            # let's add the non torch metadata like label, prompt, lie, etc
            k = i*batch_size + j
            info = ds_p_subset[k]
            
            yield dict(
                # int16 makes our storage much smaller
                hs0=float_to_int16(torch.from_numpy(hs0['hidden_states'][j])),
                scores0=hs0["scores"][j],
                grads_mlp0=hs0['grads_mlp'][j],
                # grads_mlp_cfc0=hs0['grads_mlp_cfc'][j],
                grads_attn0=hs0['grads_attn'][j],
                
                # hs1=float_to_int16(torch.from_numpy(hs1['hidden_states'][j])),
                # scores1=hs1["scores"][j],                    
                
                ds_index=index[j],
                
                **info
            )


# def md5hash(s: bytes) -> str:
#     return hashlib.md5(s).hexdigest()

# # unique hash
# def get_unique_config_hash(cfg, ds_name, split_type):
#     """
#     generates a unique name
    
#     datasets would do this use the generation kwargs but this way we have control and can handle non-picklable models and thing like the output of prompt functions if they change
    
#     # """
#     example_prompt1 = prompt_fn("text", response=0, lie=True)
#     model_repo = model.config._name_or_path
    
#     kwargs = [str(model), str(tokenizer), str(data), str(prompt_fn.__name__), N]
#     key = pickle.dumps(kwargs, 1)
#     hsh = md5hash(key)[:6]

#     sanitize = lambda s:s.replace('/', '').replace('-', '_') if s is not None else s
#     # config_name = f"{sanitize(model_repo)}-N_{N}-ns-{hsh}"
    
#     info_kwargs = dict(model_repo=model_repo, config=model.config, data=str(data), prompt_fn=str(prompt_fn.__name__), N=N, 
#                        example_prompt1=example_prompt1, 
#                        hsh=hsh)
    
#     return hsh, info_kwargs

# sanitize = lambda s:s.replace('/', '').replace('_', '-') if s is not None else s

# def ds_params2fname(dataset_params: dict) -> str:
#     prompt = sanitize(dataset_params['prompt_fmt'].__name__)
#     model_repo = sanitize(dataset_params['model_repo'].split('/')[-1])
#     dataset_name = sanitize(dataset_params['dataset_name'])
#     N = dataset_params['N']
#     N_SHOTS = dataset_params['N_SHOTS']
#     return f"model-{model_repo}_ds-{dataset_name}_{prompt}_N{N}_{N_SHOTS}shots_"
