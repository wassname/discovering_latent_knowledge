"""
Some tools modified from honest_llama.

https://github.com/likenneth/honest_llama/blob/master/utils.py#L645
"""
import numpy as np
from typing import List, Tuple, Dict, Any, Union, NewType
from einops import rearrange, reduce, repeat, asnumpy, parse_shape
import torch
import pickle
from src.config import root_folder
from src.prompts.prompt_loading import load_preproc_dataset
from transformers import AutoTokenizer, pipeline
from loguru import logger

Activations = NewType("Activations", Dict[str, torch.Tensor])

InterventionDict = NewType('InterventionDict', Dict[str, List[Tuple[np.ndarray, float]]])


def intervene(output, activation):
    # TODO need attention mask
    assert output.ndim == 3, f"expected output to be (batch, seq, vocab), got {output.shape}"
    return output + activation.to(output.device)[None, None, :]

def intervention_meta_fn2(
    outputs: torch.Tensor, layer_name: str, activations: Activations
) -> torch.Tensor:
    """see
    - honest_llama: https://github.com/likenneth/honest_llama/blob/e010f82bfbeaa4326cef8493b0dd5b8b14c6da67/validation/validate_2fold.py#L114
    - baukit: https://github.com/davidbau/baukit/blob/main/baukit/nethook.py#L42C1-L45C56

    Usage:
        edit_output = partial(intervention_meta_fn2, activations=activations)
        with TraceDict(model, layers_to_intervene, edit_output=edit_output) as ret:
            ...

    """
    if type(outputs) is tuple:
        output0 = intervene(outputs[0], activations[layer_name])
        return tuple(output0, *outputs[1:])
    elif type(outputs) is torch.Tensor:
        return intervene(outputs, activations[layer_name])
    else:
        raise ValueError(f"outputs must be tuple or tensor, got {type(outputs)}")
    
    

# def get_magnitude(activations: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
#     """
#     get center of mass direction and magnitude per layer and head
    
#     refactored to from https://github.com/likenneth/honest_llama/blob/e010f82bfbeaa4326cef8493b0dd5b8b14c6da67/utils.py#L698
#     to use einops and vector ops instead of for loop
#     """    
#     # batch length hidden_dim
#     # TODO: maybe I should just get COM for last token instead?
#     true_mass_mean = reduce(activations[labels], 'b l d -> l d', 'mean')
#     false_mass_mean = reduce(activations[~labels], 'b l d -> l d', 'mean')
#     direction = true_mass_mean - false_mass_mean
#     direction = direction / np.linalg.norm(direction, axis=1, keepdims=True) # sq norm per layer
#     activations = reduce(activations, ' b l d -> l d', 'mean')
#     proj_vals = activations * direction
#     proj_val_std = reduce(proj_vals, 'l d -> l', np.std)
#     return direction, proj_val_std


# def get_interventions_dict(activations:np.ndarray, labels: np.ndarray, layer_names: List[str]) -> InterventionDict:
#     """
#     Make an intervention dict that works with baukit.TraceDict's edit_output.

#     see https://github.com/davidbau/baukit/blob/main/baukit/nethook.py#L42C1-L45C56
#     """
#     direction, proj_val_std = get_magnitude(activations, labels)
#     out = InterventionDict({l:[] for l in layer_names})
#     for layer_i, ln in enumerate(layer_names):
#         out[ln].append((direction[layer_i].squeeze(), proj_val_std[layer_i]))
#     return out

# def intervention_meta_fn(outputs: torch.Tensor, layer_name:str, interventions: InterventionDict, alpha = 15) -> torch.Tensor:
#     """see 
#     - honest_llama: https://github.com/likenneth/honest_llama/blob/e010f82bfbeaa4326cef8493b0dd5b8b14c6da67/validation/validate_2fold.py#L114
#     - baukit: https://github.com/davidbau/baukit/blob/main/baukit/nethook.py#L42C1-L45C56
    
#     Usage:
#         intervention_fn = partial(intervention_meta_fn, interventions=interventions)
#         with TraceDict(model, layers_to_intervene, edit_output=intervention_fn) as ret:
#             ...
    
#     """
#     if type(outputs) is tuple:
#         # head_output
#         output = outputs[0]
#     elif type(outputs) is torch.Tensor:
#         output = outputs
#     else:
#         raise ValueError(f"outputs must be tuple or tensor, got {type(outputs)}")
        
#     for direction, proj_val_std in interventions[layer_name]:
#         # head_output: (batch_size, seq_len, layer_size)
#         output[:, :, :] += torch.from_numpy(alpha * proj_val_std * direction).to(output.device)[None, None, :]
#     if type(outputs) is tuple:
#         return tuple([output, *outputs[1:]])
#     else:
#         return output



def create_cache_interventions(model, tokenizer, cfg, N_fit_examples=20, batch_size=2, rep_token = -1, n_difference = 1, direction_method = 'pca'):
    """
    We want one set of interventions per model
    
    So we always load a cached version if possible. to make it approx repeatable use the same dataset etc
    """
    tokenizer_args=dict(padding="max_length", max_length=cfg.max_length, truncation=True, add_special_tokens=True)
    
    model_name = cfg.model.replace('/', '-')
    intervention_f = root_folder / 'data' / 'interventions' / f'{model_name}.pkl'
    intervention_f.parent.mkdir(exist_ok=True, parents=True)
    if not intervention_f.exists():        
        
        hidden_layers = list(range(cfg.layer_padding, model.config.num_hidden_layers, cfg.layer_stride))
        
        dataset_fit = load_preproc_dataset('imdb', tokenizer, N=N_fit_examples, seed=cfg.seed, num_shots=cfg.num_shots, max_length=cfg.max_length, prompt_format=cfg.prompt_format)
        
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
