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
from transformers import AutoTokenizer, pipeline, Pipeline
from loguru import logger

Activations = NewType("Activations", Dict[str, torch.Tensor])

InterventionDict = NewType('InterventionDict', Dict[str, List[Tuple[np.ndarray, float]]])


def intervene(output, activation):
    # TODO need attention mask
    assert output.ndim == 3, f"expected output to be (batch, seq, vocab), got {output.shape}"
    # assert torch.isfinite(output).all(), 'model output nan'
    output2 = output + activation.to(output.device)[None, None, :]
    # assert torch.isfinite(output2).all(), 'intervention lead to nan'
    return output2

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


def create_cache_interventions(model, tokenizer, cfg, N_fit_examples=20, batch_size=2, rep_token = -1, n_difference = 1, direction_method = 'pca', get_negative=False):
    """
    We want one set of interventions per model
    
    So we always load a cached version if possible. to make it approx repeatable use the same dataset etc
    """
    tokenizer_args=dict(padding="max_length", max_length=cfg.max_length, truncation=True, add_special_tokens=True)
    
    model_name = cfg.model.replace('/', '-')
    intervention_f = root_folder / 'data' / 'interventions' / f'{model_name}_{"-" if get_negative else "+"}.pkl'
    intervention_f.parent.mkdir(exist_ok=True, parents=True)
    if not intervention_f.exists():        
        
        hidden_layers = list(range(cfg.layer_padding, model.config.num_hidden_layers, cfg.layer_stride))
        
        dataset_fit = load_preproc_dataset('imdb', tokenizer, N=N_fit_examples, seed=cfg.seed, num_shots=cfg.num_shots, max_length=cfg.max_length, prompt_format=cfg.prompt_format)
        
        train_labels = np.array(dataset_fit['label_true'])
        if get_negative:
            train_labels = -1 * train_labels
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



def intervention_metrics(control_outputs_neg, baseline_outputs, control_outputs):
    signs = [-1, 0, 1]
    for i in range(len(baseline_outputs)):
        ranked = []
        
        for j, r in enumerate([control_outputs_neg, baseline_outputs, control_outputs]):        
            choices = r[i]['answer_choices']
            label = r[i]['label_true']
            ans = r[i]['ans']
            sign = signs[j]
            ranked.append(ans)
            choice_true = choices[label]
            if label==0:
                ans *= -1            
            print(f"==== Control ({signs[j]}) ====")
            print(f"Score: {ans:02.2%} of true ans `{choice_true}`")
            # print(f"Text ans: {r['text_ans'][i]}") 
        
        is_ranked = (np.argsort(ranked)==np.arange(3)).all()
        print(f"Ranked? {is_ranked} {ranked}")
        print()

def test_intervention_quality(dataset_train, activations, model, rep_control_pipeline2, batch_size=2):
    # TODO: this have bugs and is not used yet
    inputs = dataset_train[:3]
    activations_neg = {k:-v for k, v in activations.items()}
    activations_none = {k:v*0 for k, v in activations.items()}
    model.eval()
    with torch.no_grad():
        baseline_outputs = rep_control_pipeline2(inputs, batch_size=batch_size, activations=activations_none)
        control_outputs = rep_control_pipeline2(inputs, activations=activations, batch_size=batch_size)
        control_outputs_neg = rep_control_pipeline2(inputs, activations=activations_neg, batch_size=batch_size)


    intervention_metrics(control_outputs_neg, baseline_outputs, control_outputs)


def get_activations_from_reader(honesty_rep_reader: Pipeline, hidden_layers: list, coeff=1, dtype=None, device=None) -> Dict[str, float]:
    """Get activations from the honesty_rep_reader"""
    
    # FIXME: coeff is a magic number. The representation_engineering repo used 8, but it seems to vary by model?

    activations = {}
    for layer in hidden_layers:
        activations[layer] = torch.tensor(coeff * honesty_rep_reader.directions[layer] * honesty_rep_reader.direction_signs[layer])
        if device:
            activations[layer] = activations[layer].to(device)
        if dtype:
            activations[layer] = activations[layer].to(dtype)

    assert torch.isfinite(torch.concat(list(activations.values()))).all()
    return activations
