"""
Some tools modified from honest_llama.

https://github.com/likenneth/honest_llama/blob/master/utils.py#L645
"""
import numpy as np
from typing import List, Tuple, Dict, Any, Union, NewType
from einops import rearrange, reduce, repeat, asnumpy, parse_shape
import torch

InterventionDict = NewType('InterventionDict', Dict[str, List[Tuple[int, np.ndarray, float]]])

# def get_com_directions(
#     num_layers: int,
#     num_heads: int,
#     head_wise_activations: np.ndarray,
#     labels: np.ndarray,
# ) -> np.ndarray:
#     """get center of mass direction's for each layer and head."""
#     assert max(labels) == 1
#     com_directions = []
#     for layer in range(num_layers):
#         for head in range(num_heads):
#             usable_idxs = range(len(head_wise_activations))
#             usable_head_wise_activations = np.concatenate(
#                 [head_wise_activations[i][:, layer, head, :] for i in usable_idxs],
#                 axis=0,
#             )
#             true_mass_mean = np.mean(usable_head_wise_activations[labels == 1], axis=0)
#             false_mass_mean = np.mean(usable_head_wise_activations[labels == 0], axis=0)
#             com_directions.append(true_mass_mean - false_mass_mean)
#     com_directions = np.array(com_directions)

#     return com_directions


# def get_magnitude(activations: np.ndarray, labels: np.ndarray) -> np.ndarray:
#     """
#     refactored to from https://github.com/likenneth/honest_llama/blob/e010f82bfbeaa4326cef8493b0dd5b8b14c6da67/utils.py#L698
#     to use einops and vector ops instead of for loop
#     """
#     true_mass_mean = reduce(activations[labels], ' b l h -> l h', 'mean')
#     false_mass_mean = reduce(activations[~labels], ' b l h -> l h', 'mean')
#     direction = true_mass_mean - false_mass_mean
#     direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
#     activations = reduce(activations, ' b l h -> l h', 'mean')
#     proj_vals = activations * direction
#     proj_val_std = reduce(proj_vals, 'l h -> l', np.std)
#     return proj_val_std


# def layer_head_to_flattened_idx(layer, head, num_heads):
#     return layer * num_heads + head



# def get_interventions_dict(
#     probes,
#     tuning_activations,
#     layer_heads: List[Tuple[int, int]],
#     num_heads: int,
#     com_directions,
# ) -> InterventionDict:
#     """
#     Make an intervention dict that works with baukit.TraceDict's edit_output.

#     see https://github.com/davidbau/baukit/blob/main/baukit/nethook.py#L42C1-L45C56
#     """

#     # init
#     interventions = InterventionDict({})
#     for layer, head in layer_heads:
#         interventions[f"model.layers.{layer}.self_attn.head_out"] = []
        
        
#     std = get_magnitude(activations, labels)
        
#     # work out magnitude of intervention, then record
#     for layer, head in layer_heads:
#         direction = com_directions[layer_head_to_flattened_idx(layer, head, num_heads)]
#         direction = direction / np.linalg.norm(direction)
#         activations = tuning_activations[:, layer, head, :]  # batch x 128
#         proj_vals = activations @ direction.T
#         proj_val_std = float(np.std(proj_vals)) # TODO check this is meant to be float
#         interventions[f"model.layers.{layer}.self_attn.head_out"].append(
#             (head, direction.squeeze(), proj_val_std)
#         )

#     # sort keys by head index
#     for layer, head in layer_heads:
#         interventions[f"model.layers.{layer}.self_attn.head_out"] = sorted(
#             interventions[f"model.layers.{layer}.self_attn.head_out"],
#             key=lambda x: x[0],
#         )

#     return interventions


def get_magnitude(layer_activations: np.ndarray, labels: np.ndarray, num_heads:int) -> Tuple[np.ndarray,np.ndarray]:
    """
    get center of mass direction and magnitude per layer and head
    
    refactored to from https://github.com/likenneth/honest_llama/blob/e010f82bfbeaa4326cef8493b0dd5b8b14c6da67/utils.py#L698
    to use einops and vector ops instead of for loop
    """
    # we intervene with statistic per head, like in honest_llama
    activations = rearrange(layer_activations, 'b l (h d) -> b l h d', h=num_heads)
    
    true_mass_mean = reduce(activations[labels], ' b l h d -> l h d', 'mean')
    false_mass_mean = reduce(activations[~labels], ' b l h d -> l h d', 'mean')
    direction = true_mass_mean - false_mass_mean
    direction = direction / np.linalg.norm(direction, axis=1, keepdims=True) # sq norm per layer
    activations = reduce(activations, ' b l h d -> l h d', 'mean')
    proj_vals = activations * direction
    proj_val_std = reduce(proj_vals, 'l h d -> l h', np.std)
    return direction, proj_val_std


def get_interventions_dict(activations:np.ndarray, labels: np.ndarray, layer_names: List[str], num_heads: int) -> InterventionDict:
    """
    Make an intervention dict that works with baukit.TraceDict's edit_output.

    see https://github.com/davidbau/baukit/blob/main/baukit/nethook.py#L42C1-L45C56
    """
    direction, proj_val_std = get_magnitude(activations, labels, num_heads)
    out = InterventionDict({l:[] for l in layer_names})
    for layer_i, ln in enumerate(layer_names):
        for head in range(num_heads):    
            out[ln].append((head, direction[layer_i, head].squeeze(), proj_val_std[layer_i, head]))
    return out

def intervention_meta_fn(head_outputs: torch.Tensor, layer_name:str, interventions: InterventionDict, num_heads, alpha = 15) -> torch.Tensor:
    """see 
    - honest_llama: https://github.com/likenneth/honest_llama/blob/e010f82bfbeaa4326cef8493b0dd5b8b14c6da67/validation/validate_2fold.py#L114
    - baukit: https://github.com/davidbau/baukit/blob/main/baukit/nethook.py#L42C1-L45C56
    
    Usage:
        intervention_fn = partial(intervention_meta_fn, interventions=interventions)
        with TraceDict(model, layers_to_intervene, edit_output=intervention_fn) as ret:
            ...
    
    """
    head_output, a, b = head_outputs
    head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads)
    for head, direction, proj_val_std in interventions[layer_name]:
        # head_output: (batch_size, seq_len, num_heads, head_size)
        head_output[:, -1:, head, :] += torch.from_numpy(alpha * proj_val_std * direction).to(head_output.device)
    head_output = rearrange(head_output, 'b s h d -> b s (h d)')
    head_outputs = (head_output, a, b)
    return head_outputs
