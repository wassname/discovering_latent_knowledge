
from tqdm.auto import tqdm
from src.datasets.hs import ExtractHiddenStates
from torch.utils.data import DataLoader
from datasets import Dataset
import hashlib
import pickle
import numpy as np

def batch_hidden_states(model, tokenizer, data: Dataset, n=100, batch_size=2, mcdropout=True):
    """
    Given an encoder-decoder model, a list of data, computes the contrast hidden states on n random examples.
    Returns numpy arrays of shape (n, hidden_dim) for each candidate label, along with a boolean numpy array of shape (n,)
    with the ground truth labels
    
    This is deliberately simple so that it's easy to understand, rather than being optimized for efficiency
    """
    ehs = ExtractHiddenStates(model, tokenizer)
    
    ds_t_subset = data.select(range(n))
    ds_t_subset.set_format(type='torch', columns=['input_ids', 'label', 'attention_mask'])
    
    ds_p_subset = data.select(range(n))
    ds_p_subset.set_format(type="pandas", columns=['lie', 'label', 'prompt', 'prompt_truncated'])
    
    dl = DataLoader(ds_t_subset, batch_size=batch_size, shuffle=True)
    for i, batch in enumerate(tqdm(dl, desc='get hidden states')):
        input_ids, true_labels, attention_mask =  batch["input_ids"], batch["label"], batch["attention_mask"]
        nn = len(input_ids)
        index = i*batch_size+np.arange(nn)
        
        # different due to dropout
        hs0 = ehs.get_batch_of_hidden_states(input_ids=input_ids, attention_mask=attention_mask, use_mcdropout=mcdropout)
        if mcdropout:
            hs1 = ehs.get_batch_of_hidden_states(input_ids=input_ids, attention_mask=attention_mask, use_mcdropout=mcdropout)
            
            # QC
            if i==0:
                eps=1e-5
                mpe = lambda x,y: np.mean(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
                a,b=hs1['hidden_states'],hs0['hidden_states']
                assert mpe(a,b)>eps, "the hidden state pairs should be different but are not. Check model.config.use_cache==False, check this model has dropout in it's arch"
        else:
            hs1 = hs0

        
        for j in range(nn):
            # let's add the non torch metadata like label, prompt, lie, etc
            k = i*batch_size + j
            info = ds_p_subset[k]
            
            yield dict(
                hs0=hs0['hidden_states'][j],
                scores1=hs0["scores"][j],
                
                hs1=hs1['hidden_states'][j],
                scores2=hs1["scores"][j],                    
                
                true=true_labels[j].item(),
                index=index[j],
                
                **info
            )


def md5hash(s: bytes) -> str:
    return hashlib.md5(s).hexdigest()

# unique hash
def get_unique_config_hash(prompt_fn, model, tokenizer, data, N):
    """
    generates a unique name
    
    datasets would do this use the generation kwargs but this way we have control and can handle non-picklable models and thing like the output of prompt functions if they change
    
    # """
    example_prompt1 = prompt_fn("text", response=0, lie=True)
    model_repo = model.config._name_or_path
    
    kwargs = [str(model), str(tokenizer), str(data), str(prompt_fn.__name__), N]
    key = pickle.dumps(kwargs, 1)
    hsh = md5hash(key)[:6]

    sanitize = lambda s:s.replace('/', '').replace('-', '_') if s is not None else s
    # config_name = f"{sanitize(model_repo)}-N_{N}-ns-{hsh}"
    
    info_kwargs = dict(model_repo=model_repo, config=model.config, data=str(data), prompt_fn=str(prompt_fn.__name__), N=N, 
                       example_prompt1=example_prompt1, 
                       hsh=hsh)
    
    return hsh, info_kwargs

sanitize = lambda s:s.replace('/', '').replace('_', '-') if s is not None else s

def ds_params2fname(dataset_params: dict) -> str:
    prompt = sanitize(dataset_params['prompt_fmt'].__name__)
    model_repo = sanitize(dataset_params['model_repo'].split('/')[-1])
    dataset_name = sanitize(dataset_params['dataset_name'])
    N = dataset_params['N']
    N_SHOTS = dataset_params['N_SHOTS']
    return f"model-{model_repo}_ds-{dataset_name}_{prompt}_N{N}_{N_SHOTS}shots_"
