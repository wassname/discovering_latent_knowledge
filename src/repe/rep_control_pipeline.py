import torch
from transformers.pipelines import (
    TextGenerationPipeline,
    FeatureExtractionPipeline,
    Pipeline,
)
from transformers.pipelines.base import GenericTensor
from .rep_control_reading_vec import WrappedReadingVecModel
from typing import Dict


class RepControlPipeline(FeatureExtractionPipeline):
    def __init__(
        self,
        model,
        tokenizer,
        layers,
        block_name="decoder_block",
        control_method="reading_vec",
        max_length=555,
        **kwargs,
    ):
        # TODO: implement different control method and supported intermediate modules for different models
        assert control_method == "reading_vec", f"{control_method} not supported yet"
        assert (
            block_name == "decoder_block"
            or "LlamaForCausalLM" in model.config.architectures
        ), f"{model.config.architectures} {block_name} not supported yet"
        self.wrapped_model = WrappedReadingVecModel(model, tokenizer)
        self.wrapped_model.unwrap()
        self.wrapped_model.wrap_block(layers, block_name=block_name)
        self.block_name = block_name
        self.layers = layers
        self.max_length = max_length

        super().__init__(model=model, tokenizer=tokenizer, **kwargs)
        
    def preprocess(self, inputs, **tokenize_kwargs) -> Dict[str, GenericTensor]:
        # tokenize a batch of inputs
        return_tensors = self.framework
        model_inputs = self.tokenizer(inputs['question'], return_tensors=return_tensors, return_attention_mask=True, add_special_tokens=True, truncation=True, padding="max_length", max_length=self.max_length, **tokenize_kwargs)
        return {**inputs, **model_inputs}

    def __call__(self, text_inputs, activations=None, **kwargs):
        if activations is not None:
            self.wrapped_model.reset()
            self.wrapped_model.set_controller(self.layers, activations, self.block_name)

        outputs = super().__call__(text_inputs, **kwargs)
        self.wrapped_model.reset()

        return outputs

    def _forward(self, model_inputs):
        inputs = dict(input_ids=model_inputs['input_ids'], attention_mask=model_inputs['attention_mask'])
        inputs.update(
            {"use_cache": False, "output_hidden_states": True, "return_dict": True}
        )
        with torch.no_grad():
            model_outputs = self.model(**inputs)
        
        # retain some of the inputs
        keep_cols = ["answer_choices", "input_ids", "attention_mask"]
        model_outputs = {**model_inputs, **model_outputs}
        return model_outputs

    def postprocess(self, o):
        # note this sometimes deals with a batch, sometimes with a single result. infuriating
        assert isinstance(o, dict) and o['logits'].ndim==3, f"expected dict with logits of shape (batch, seq, vocab), got {o['logits'].shape}"
        # This is called once for each result, but the text pipeline is set up to hande multiple...
        # This is called once for each result, but the text pipeline is set up to hande multiple...
        o["end_logits"] = o["logits"][:, -1, :].float()
        # hidden_states = list(o.hidden_states)
        o["input_truncated"] = self.tokenizer.batch_decode(o['input_ids'])
        o["truncated"] = torch.sum(o["attention_mask"], 1)==self.max_length
        o["text_ans"] = self.tokenizer.batch_decode(o["end_logits"].argmax(-1))
        o['choice_ids'] = row_choice_ids(o, self.tokenizer)
        return o


from typing import List, Tuple, Dict, Any, Union, NewType
from baukit.nethook import Trace, TraceDict, recursive_copy
from src.datasets.intervene import InterventionDict, intervention_meta_fn
from functools import partial
from src.datasets.scores import choice2ids

Activations = NewType("InterventionDict", Dict[str, torch.Tensor])

def row_choice_ids(answer_choices, tokenizer):
    return choice2ids([[c] for c in answer_choices], tokenizer)

def intervention_meta_fn2(
    output: torch.Tensor, layer_name: str, activations: Activations
) -> torch.Tensor:
    """see
    - honest_llama: https://github.com/likenneth/honest_llama/blob/e010f82bfbeaa4326cef8493b0dd5b8b14c6da67/validation/validate_2fold.py#L114
    - baukit: https://github.com/davidbau/baukit/blob/main/baukit/nethook.py#L42C1-L45C56

    Usage:
        edit_output = partial(intervention_meta_fn2, activations=activations)
        with TraceDict(model, layers_to_intervene, edit_output=edit_output) as ret:
            ...

    """
    for activation in activations[layer_name]:
        # TODO might be model specific?
        output[:, :, :] += torch.from_numpy(activation).to(output.device)[None, None, :]

    return output


class RepControlPipeline2(FeatureExtractionPipeline):
    """This version uses baukit."""
    def __init__(self, model, tokenizer, max_length, **kwargs):
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)
        self.max_length = max_length

    def __call__(self, model_inputs, activations=None, **kwargs):
        if activations is not None:
            # FIXME model specific
            layers_names = [f'model.model.layers.{i}.post_attention_layernorm' for i in activations.keys()]
            edit_fn = partial(intervention_meta_fn2, activations=activations)
            with TraceDict(
                self.model, layers_names, detach=True, edit_output=edit_fn
            ) as ret:
                outputs = super().__call__(model_inputs, **kwargs)
        else:
            outputs = super().__call__(model_inputs, **kwargs)
        return outputs
    
    def preprocess(self, inputs, **tokenize_kwargs) -> Dict[str, GenericTensor]:
        # tokenize a batch of inputs
        return_tensors = self.framework
        model_inputs = self.tokenizer(inputs['question'], return_tensors=return_tensors, return_attention_mask=True, add_special_tokens=True, truncation=True, padding="max_length", max_length=self.max_length, **tokenize_kwargs)
        return {**inputs, **model_inputs}

    def _forward(self, model_inputs):
        inputs = dict(input_ids=model_inputs['input_ids'], attention_mask=model_inputs['attention_mask'])
        inputs.update(
            {"use_cache": False, "output_hidden_states": True, "return_dict": True}
        )
        with torch.no_grad():
            model_outputs = self.model(**inputs)
        
        # retain some of the inputs
        keep_cols = ["answer_choices", "input_ids", "attention_mask"]
        model_outputs = {**model_inputs, **model_outputs}
        return model_outputs

    def postprocess(self, o):
        # note this sometimes deals with a batch, sometimes with a single result. infuriating
        assert isinstance(o, dict) and o['logits'].ndim==3, f"expected dict with logits of shape (batch, seq, vocab), got {o['logits'].shape}"
        # This is called once for each result, but the text pipeline is set up to hande multiple...
        # This is called once for each result, but the text pipeline is set up to hande multiple...
        o["end_logits"] = o["logits"][:, -1, :].float()
        # hidden_states = list(o.hidden_states)
        o["input_truncated"] = self.tokenizer.batch_decode(o['input_ids'])
        o["truncated"] = torch.sum(o["attention_mask"], 1)==self.max_length
        o["text_ans"] = self.tokenizer.batch_decode(o["end_logits"].argmax(-1))
        o['choice_ids'] = row_choice_ids(o, self.tokenizer)
        return o
