import re
import torch
from transformers.pipelines import (
    TextGenerationPipeline,
    FeatureExtractionPipeline,
    Pipeline,
)
from transformers.pipelines.base import GenericTensor
from datasets import Dataset
from typing import List, Tuple, Dict, Any, Union, NewType
from baukit.nethook import Trace, TraceDict, recursive_copy
from functools import partial
from einops import rearrange
from transformers.modeling_outputs import ModelOutput
from src.datasets.scores import choice2ids, default_class2choices, scores2choice_probs2
# from src.datasets.scores import scores2choice_probs
from src.helpers.torch import clear_mem, detachcpu

Activations = NewType("Activations", Dict[str, torch.Tensor])


def hacky_sanitize_outputs(o):
    """I can't find the mem leak, so lets just detach, cpu, clone, freemem."""
    o = {k: detachcpu(v) for k, v in o.items()}
    o = recursive_copy(o, detach=True, clone=True)
    clear_mem()
    return o

def row_choice_ids(answer_choices, tokenizer):
    return choice2ids([c for c in answer_choices], tokenizer)


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
    
    
# def split_outputs(o):


class RepControlPipeline2(FeatureExtractionPipeline):
    """This version uses baukit."""
    def __init__(self, model, tokenizer, max_length, layer_name_tmpl="model.layers.{}", **kwargs):
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)
        self.max_length = max_length
        self.layer_name_tmpl = layer_name_tmpl
        
        # self.default_class2choiceids = choice2ids(default_class2choices, tokenizer)

    def __call__(self, model_inputs, **kwargs):
        return super().__call__(model_inputs, **kwargs)
    
    def _sanitize_parameters(self, truncation=None, tokenize_kwargs=None, return_tensors=None, activations=None, **kwargs):
        """This processed the init params."""
        if tokenize_kwargs is None:
            tokenize_kwargs = {}

        preprocess_params = tokenize_kwargs

        forward_params = {'activations': activations}
        
        postprocess_params = {}
        if return_tensors is not None:
            postprocess_params["return_tensors"] = return_tensors

        return preprocess_params, forward_params, postprocess_params
    
    def preprocess(self, inputs: dict, **tokenize_kwargs) -> Dict[str, GenericTensor]:
        # tokenize a batch of inputs
        return_tensors = self.framework
        
        # if the pipeline is in "single mode or "generator mode" it gets singles, which we turn into a batch. In generator mode the batches of single items will get concatenated fine
        if isinstance(inputs['question'], str):
            inputs = {k: [v] for k, v in inputs.items()}
            
        # tokenize if needed
        if 'input_ids' not in inputs:
            model_inputs = self.tokenizer(inputs['question'], return_tensors=return_tensors, return_attention_mask=True, add_special_tokens=True, truncation=True, padding="max_length", max_length=self.max_length, **tokenize_kwargs)
            inputs = {**inputs, **model_inputs}
        
        # to device
        inputs["input_ids"] = torch.tensor(inputs['input_ids'], dtype=torch.long, device=self.model.device)
        inputs["attention_mask"] = torch.tensor(inputs['attention_mask'], dtype=torch.bool, device=self.model.device)
        return inputs

    def _forward(self, inputs, activations) -> ModelOutput:
        assert inputs['input_ids'].ndim == 2, f"expected input_ids to be (batch, seq), got {inputs['input_ids'].shape}"

        # make intervention functions
        layers_names = [self.layer_name_tmpl.format(i) for i in activations.keys()]                
        activations_pos_i = Activations({self.layer_name_tmpl.format(k):v for k,v in activations.items()})
        activations_neg_i = Activations({self.layer_name_tmpl.format(k):-v for k,v in activations.items()})
        edit_fn_pos = partial(intervention_meta_fn2, activations=activations_pos_i)
        edit_fn_neg = partial(intervention_meta_fn2, activations=activations_neg_i)
        
        self.model.eval()
        model_in = dict(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            use_cache=False,
            output_hidden_states=True,
            return_dict=True
        )
        
        def transform_model_output(o):
            # hidden states come at as lists of layers, lets stack them
            o['hidden_states'] = rearrange(list(o['hidden_states']), 'l b t h -> b l t h')
            
            # we only want the last token
            o = ModelOutput(end_hidden_states=o['hidden_states'][:, :, -1], end_logits=o['logits'][:, -1])
            
            return o
        
        # intervent in the negative and positive direction
        with torch.no_grad():
            with TraceDict(
                self.model, layers_names, detach=True, edit_output=edit_fn_pos
            ) as ret:
                outputs_pos = transform_model_output(self.model(**model_in))
                
            with TraceDict(
                self.model, layers_names, detach=True, edit_output=edit_fn_neg
            ) as ret:
                outputs_neg = transform_model_output(self.model(**model_in))
                
        # stack the outputs
        o = {k: torch.stack([outputs_neg[k], outputs_pos[k]], -1) for k in outputs_neg.keys()}
        
        # batch of outputs and inputs. retain some of the inputs
        return ModelOutput(**o, **inputs)

    def postprocess(self, o: ModelOutput):
        o = hacky_sanitize_outputs(o)
        # note this sometimes deals with a batch, sometimes with a single result. infuriating
        res = []
        for i in range(len(o['input_ids'])):
            o_i = {k: v[i] for k, v in o.items()}
            res.append(self.postprocess_single(o_i))
            
        # it seems to expect us to squeeze single results
        if len(res)==1:
            return res[0]
        else:
            return res
        
    def postprocess_single(self, o: dict) -> dict:
        assert isinstance(o, dict) and o['end_logits'].ndim==2, f"expected dict with logits of shape (seq, vocab), got {o['end_logits'].shape}"
        # assert o['logits'].shape[0]==1, f"postprocess expected batch size 1, got {o['logits'].shape[0]}"
        
        input_ids = torch.tensor(o['input_ids'], dtype=torch.long)
        o["input_truncated"] = self.tokenizer.decode(input_ids)
        
        o["truncated"] = torch.tensor(o["attention_mask"]).sum()==self.max_length
        o["text_ans"] = self.tokenizer.decode(o["end_logits"].argmax(-1))
        
        if 'answer_choices' in o:
            answer_choices = o['answer_choices']
        else:
            answer_choices = default_class2choices
        
        o['choice_ids'] = row_choice_ids(answer_choices, self.tokenizer)
        
        ii = o['end_logits'].shape[1]
        p = o['add_ans'] = torch.stack([scores2choice_probs2(o['end_logits'][:, i], o['choice_ids']) for i in range(ii)], 1)
        o['ans'] = p[1] / (torch.sum(p, 0) + 1e-5)
        
        
        # lets delete all the large arrays we don't need. We don't need anything with 3 dims, as we only need the things from the last token
        for k in ['input_ids', 'attention_mask', 'logits', 'hidden_states']:
            if k in o:
                del o[k]
        
        # ah to make a dataset we need to return one at a time, right now it's Dict[str, Batch]. e.g. hiddenstates={layer_1:[2, 555, 5120]....
        o = hacky_sanitize_outputs(o)
        # {k:v.shape for k,v in o.items() if hasattr(v, 'shape')}
        return o

