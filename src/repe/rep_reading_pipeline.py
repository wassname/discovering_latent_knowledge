from typing import List, Union, Optional
from transformers import Pipeline
import torch
import numpy as np

from .interventions import DIRECTION_FINDERS, Intervention, LayerInterventions

class RepReadingPipeline(Pipeline):
    """Returns the directions for each layer, for each example."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_hidden_states(
            self, 
            outputs,
            rep_token: Union[str, int]=-1,
            hidden_layers: Union[List[int], int]=-1,
            which_hidden_states: Optional[str]=None):
        
        if hasattr(outputs, 'encoder_hidden_states') and hasattr(outputs, 'decoder_hidden_states'):
            outputs['hidden_states'] = outputs[f'{which_hidden_states}_hidden_states']
    
        hidden_states_layers = {}
        for layer in hidden_layers:
            hidden_states = outputs['hidden_states'][layer]
            hidden_states =  hidden_states[:, rep_token, :]
            hidden_states_layers[layer] = hidden_states.cpu().to(dtype=torch.float32).detach()

        return hidden_states_layers

    def _sanitize_parameters(self, 
                             intervention: Intervention=None,
                             rep_token: Union[str, int]=-1,
                             hidden_layers: Union[List[int], int]=-1,
                             component_index: int=0,
                             which_hidden_states: Optional[str]=None,
                             **tokenizer_kwargs):
        preprocess_params = tokenizer_kwargs
        forward_params =  {}
        postprocess_params = {}

        forward_params['rep_token'] = rep_token

        if not isinstance(hidden_layers, list):
            hidden_layers = [hidden_layers]


        assert intervention is None or len(intervention.directions) == len(hidden_layers), f"expect total intervention directions ({len(intervention.directions)})== total hidden_layers ({len(hidden_layers)})"                 
        forward_params['intervention'] = intervention
        forward_params['hidden_layers'] = hidden_layers
        forward_params['component_index'] = component_index
        forward_params['which_hidden_states'] = which_hidden_states
        
        return preprocess_params, forward_params, postprocess_params
 
    def preprocess(
            self, 
            inputs: Union[str, List[str], List[List[str]]],
            **tokenizer_kwargs):

        if self.image_processor:
            return self.image_processor(inputs, add_end_of_utterance_token=False, return_tensors="pt")
        return self.tokenizer(inputs, return_tensors=self.framework, **tokenizer_kwargs)

    def postprocess(self, outputs):
        return outputs

    def _forward(self, model_inputs, rep_token, hidden_layers, intervention=None, component_index=0, which_hidden_states=None):
        """
        Args:
        - which_hidden_states (str): Specifies which part of the model (encoder, decoder, or both) to compute the hidden states from. 
                        It's applicable only for encoder-decoder models. Valid values: 'encoder', 'decoder'.
        """
        # get model hidden states and optionally transform them with a RepReader
        with torch.no_grad():
            if hasattr(self.model, "encoder") and hasattr(self.model, "decoder"):
                decoder_start_token = [self.tokenizer.pad_token] * model_inputs['input_ids'].size(0)
                decoder_input = self.tokenizer(decoder_start_token, return_tensors="pt").input_ids
                model_inputs['decoder_input_ids'] = decoder_input
            outputs =  self.model(**model_inputs, output_hidden_states=True)
        hidden_states = self._get_hidden_states(outputs, rep_token, hidden_layers, which_hidden_states)
        
        if intervention is None:
            return hidden_states
        
        return intervention(hidden_states, hidden_layers, component_index)


    def _batched_string_to_hiddens(self, train_inputs, rep_token, hidden_layers, batch_size, which_hidden_states, **tokenizer_args):
        # Wrapper method to get a dictionary hidden states from a list of strings
        hidden_states_outputs = self(train_inputs, rep_token=rep_token,
            hidden_layers=hidden_layers, batch_size=batch_size, intervention=None, which_hidden_states=which_hidden_states, **tokenizer_args)
        # return hidden_states_outputs
        hidden_states = {layer: [] for layer in hidden_layers}
        for hidden_states_batch in hidden_states_outputs:
            for layer in hidden_states_batch:
                hidden_states[layer].extend(hidden_states_batch[layer])
        for layer in hidden_layers:
            hidden_states[layer] = torch.stack(hidden_states[layer])
        return hidden_states

    def get_directions(
            self, 
            train_inputs: Union[str, List[str], List[List[str]]], 
            train_labels: List[int],
            rep_token: Union[str, int]=-1, 
            hidden_layers: Union[str, int]=-1,
            n_difference: int = 1,
            batch_size: int = 8, 
            direction_method: str = 'mm',
            direction_finder_kwargs: dict = {},
            which_hidden_states: Optional[str]=None,
            layer_name_tmpl: str = "model.layers.{}",
            **tokenizer_args,):
        """Train a RepReader on the training data.
        Args:
            batch_size: batch size to use when getting hidden states
            direction_method: string specifying the RepReader strategy for finding directions
            direction_finder_kwargs: kwargs to pass to RepReader constructor
        """

        if not isinstance(hidden_layers, list): 
            assert isinstance(hidden_layers, int)
            hidden_layers = [hidden_layers]

        # initialize a DirectionFinder
        Intervention = DIRECTION_FINDERS[direction_method]

        # get raw hidden states for the train inputs
        hidden_states = self._batched_string_to_hiddens(train_inputs, rep_token, hidden_layers, batch_size, which_hidden_states, **tokenizer_args)
        
        # get differences between pairs
        relative_hidden_states = {k: torch.clone(v) for k, v in hidden_states.items()}
        for layer in hidden_layers[1:]:
            for _ in range(n_difference):
                relative_hidden_states[layer] = relative_hidden_states[layer] - relative_hidden_states[layer-1]

		# fit probe
        # TODO: use rel or abs?
        probe = LayerInterventions.from_data(Intervention, relative_hidden_states, torch.LongTensor(train_labels), layer_name_tmpl)
        
        return probe
