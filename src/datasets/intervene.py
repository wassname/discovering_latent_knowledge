"""
Some tools modified from honest_llama.

https://github.com/likenneth/honest_llama/blob/master/utils.py#L645
"""
import numpy as np
from typing import List, Tuple, Dict, Any, Union, NewType
from einops import rearrange, reduce, repeat, asnumpy, parse_shape
import torch

InterventionDict = NewType('InterventionDict', Dict[str, List[Tuple[np.ndarray, float]]])


def get_magnitude(activations: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    """
    get center of mass direction and magnitude per layer and head
    
    refactored to from https://github.com/likenneth/honest_llama/blob/e010f82bfbeaa4326cef8493b0dd5b8b14c6da67/utils.py#L698
    to use einops and vector ops instead of for loop
    """    
    true_mass_mean = reduce(activations[labels], ' b l d -> l d', 'mean')
    false_mass_mean = reduce(activations[~labels], ' b l d -> l d', 'mean')
    direction = true_mass_mean - false_mass_mean
    direction = direction / np.linalg.norm(direction, axis=1, keepdims=True) # sq norm per layer
    activations = reduce(activations, ' b l d -> l d', 'mean')
    proj_vals = activations * direction
    proj_val_std = reduce(proj_vals, 'l d -> l', np.std)
    return direction, proj_val_std


def get_interventions_dict(activations:np.ndarray, labels: np.ndarray, layer_names: List[str]) -> InterventionDict:
    """
    Make an intervention dict that works with baukit.TraceDict's edit_output.

    see https://github.com/davidbau/baukit/blob/main/baukit/nethook.py#L42C1-L45C56
    """
    direction, proj_val_std = get_magnitude(activations, labels)
    out = InterventionDict({l:[] for l in layer_names})
    for layer_i, ln in enumerate(layer_names):
        out[ln].append((direction[layer_i].squeeze(), proj_val_std[layer_i]))
    return out

def intervention_meta_fn(outputs: torch.Tensor, layer_name:str, interventions: InterventionDict, alpha = 15) -> torch.Tensor:
    """see 
    - honest_llama: https://github.com/likenneth/honest_llama/blob/e010f82bfbeaa4326cef8493b0dd5b8b14c6da67/validation/validate_2fold.py#L114
    - baukit: https://github.com/davidbau/baukit/blob/main/baukit/nethook.py#L42C1-L45C56
    
    Usage:
        intervention_fn = partial(intervention_meta_fn, interventions=interventions)
        with TraceDict(model, layers_to_intervene, edit_output=intervention_fn) as ret:
            ...
    
    """
    output, a, b = outputs
    for direction, proj_val_std in interventions[layer_name]:
        # head_output: (batch_size, seq_len, layer_size)
        output[:, -1:, :] += torch.from_numpy(alpha * proj_val_std * direction).to(output.device)
    outputs = (output, a, b)
    return outputs
