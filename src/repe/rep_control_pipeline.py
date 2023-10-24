import torch
from transformers.pipelines import (
    TextGenerationPipeline,
    FeatureExtractionPipeline,
    Pipeline,
)
from transformers.pipelines.base import GenericTensor
from .rep_control_reading_vec import WrappedReadingVecModel
from typing import Dict
from .rep_control_pipeline_baukit import RepControlPipeline2



class RepControlPipeline(RepControlPipeline2):
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

        super().__init__(model=model, tokenizer=tokenizer, max_length=max_length, **kwargs)
        
    def __call__(self, text_inputs, activations=None, **kwargs):
        if activations is not None:
            self.wrapped_model.reset()
            self.wrapped_model.set_controller(self.layers, activations, self.block_name)

        outputs = super().__call__(text_inputs, **kwargs)
        self.wrapped_model.reset()

        return outputs
