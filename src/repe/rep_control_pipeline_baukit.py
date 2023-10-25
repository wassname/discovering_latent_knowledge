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

    def __call__(self, model_inputs, activations=None, **kwargs):
        with torch.no_grad():
            if activations is not None:
                activations_i = Activations({self.layer_name_tmpl.format(k):v for k,v in activations.items()})
                layers_names = [self.layer_name_tmpl.format(i) for i in activations.keys()]
                edit_fn = partial(intervention_meta_fn2, activations=activations_i)
                with TraceDict(
                    self.model, layers_names, detach=True, edit_output=edit_fn
                ) as ret:
                    outputs = super().__call__(model_inputs, **kwargs)
            else:
                outputs = super().__call__(model_inputs, **kwargs)
        return outputs
    
    def preprocess(self, inputs: Dataset, **tokenize_kwargs) -> Dict[str, GenericTensor]:
        # tokenize a batch of inputs
        return_tensors = self.framework
        
        # if the pipeline is in "single mode", turn it into a batch
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

    def _forward(self, model_inputs):
        
        assert model_inputs['input_ids'].ndim == 2, f"expected input_ids to be (batch, seq), got {model_inputs['input_ids'].shape}"
        
        self.model.eval()
        inputs = dict(
            input_ids=model_inputs['input_ids'],
            attention_mask=model_inputs['attention_mask'],
            use_cache=False,
            output_hidden_states=True,
            return_dict=True
        )
        with torch.no_grad():
            model_outputs = self.model(**inputs)
        # o = {k: detachcpu(v) for k, v in o.items()}
        # o = recursive_copy(o)
        # clear_mem()
        
        # hidden states come at as lists of layers, lets concat them
        model_outputs['hidden_states'] = rearrange(list(model_outputs['hidden_states']), 'l b t h -> b l t h')
        
        # batch of outputs and inputs. retain some of the inputs
        model_outputs = {**model_inputs, **model_outputs}
        return hacky_sanitize_outputs(model_outputs)

    def postprocess(self, o):
        # note this sometimes deals with a batch, sometimes with a single result. infuriating
        # TODO loop through results and yeild them one at a time
        for i in range(len(o['input_ids'])):
            o_i = {k: v[i] for k, v in o.items()}
            o_i = self.postprocess1(o_i)
            yield o_i
        
    def postprocess1(self, o):
        assert isinstance(o, dict) and o['logits'].ndim==2, f"expected dict with logits of shape (seq, vocab), got {o['logits'].shape}"
        # assert o['logits'].shape[0]==1, f"postprocess expected batch size 1, got {o['logits'].shape[0]}"
        # This is called once for each result, but the text pipeline is set up to hande multiple...
        
        # TODO for k in ks: o[k] = o[k].squeeze(0)
        
        o["end_logits"] = o["logits"][-1, :].float()
        
        input_ids = torch.tensor(o['input_ids'], dtype=torch.long)
        o["input_truncated"] = self.tokenizer.decode(input_ids)
        
        o["truncated"] = torch.tensor(o["attention_mask"]).sum()==self.max_length
        o["text_ans"] = self.tokenizer.decode(o["end_logits"].argmax(-1))
        
        if 'answer_choices' in o:
            answer_choices = o['answer_choices']
            # if isinstance(answer_choices[0][0], str):
                # answer_choices = [answer_choices]
        else:
            answer_choices = default_class2choices # self.default_class2choiceids
        
        o['choice_ids'] = row_choice_ids(answer_choices, self.tokenizer)
        
        p = o['add_ans'] = scores2choice_probs2(o['end_logits'], o['choice_ids']) 
        o['ans'] = p[1] / (torch.sum(p) + 1e-5)
        
        o = hacky_sanitize_outputs(o)
        
        # ah to make a dataset we need to return one at a time, right now it's Dict[str, Batch]. e.g. hiddenstates={layer_1:[2, 555, 5120]....
        return o

