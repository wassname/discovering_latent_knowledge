from dataclasses import dataclass
import lightning as pl
import torch
from loguru import logger
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    AutoConfig,
    AutoModel,
    PreTrainedTokenizer,
    PreTrainedModel
)
from typing import Optional, List, Tuple, Dict
from transformers import LogitsProcessorList

from src.helpers.torch import to_numpy
from src.datasets.dropout import enable_dropout


from tqdm.auto import tqdm
# from src.datasets.hs import ExtractHiddenStates
from torch.utils.data import DataLoader
from datasets import Dataset
import numpy as np

default_class2choices = {False: ['No', 'Negative', 'no', 'false', 'wrong'], True: ['Yes', 'Positive', 'yes', 'true', 'correct', 'right']}


def get_choices_as_tokens(
    tokenizer, choices:List[str] = ["Positive"], whitespace_first=True
) -> Tuple[List[int], List[int]]:
    
    # Note some tokenizers differentiate between "no", "\nno", so we sometime need to add whitespace beforehand...
    if not whitespace_first:
        raise NotImplementedError('TODO')
    
    ids = []
    for c in choices:
        id_ = tokenizer(f"\n{c}", add_special_tokens=True)["input_ids"][-1]
        ids.append(id_)
        
        c2 = tokenizer.decode([id_])
        assert tokenizer.decode([id_]) == c, f'tokenizer.decode(tokenizer(`{c}`))==`{c2}`!=`{c}`'

    return ids


@dataclass
class ExtractHiddenStates:
    
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    layer_stride: int = 1
    layer_padding: int = 2


    def get_batch_of_hidden_states(
        self,
        input_text: Optional[List[str]] = None,
        input_ids: torch.Tensor = None,
        truncation_length=999,
        output_attentions=False,
        use_mcdropout=True,
        debug=False,
    ):
        """
        Given a decoder model and a batch of texts, gets a pair of hidden states (in a given layer) on that input texts
        """
        assert (input_ids is not None) or (input_text is not None), "need to provide input_ids or input_text"
        assert self.tokenizer.truncation_side == 'left'
        
        if input_text:
            input_ids = self.tokenizer(
                input_text,
                return_tensors="pt",
                add_special_tokens=True,
                padding='max_length', max_length=truncation_length, truncation=True
            ).input_ids.to(self.model.device)

        # forward pass
        last_token = -1
        with torch.no_grad():
            input_ids = input_ids.to(self.model.device)
            self.model.eval()
            if use_mcdropout:
                enable_dropout(self.model, use_mcdropout)

            # Forward for one step is the same as greedy generation for one step
            # https://github.com/huggingface/transformers/blob/234cfefbb083d2614a55f6093b0badfb2efc3b45/src/transformers/generation_utils.py#L1528
            outputs = self.model.forward(
                input_ids,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=True,
                use_cache=False,
            )

            outputs["scores"] = outputs.logits[:, last_token, :]

            layers = self.get_layer_selection(outputs)
            
            attentions = None
            if output_attentions:
                attentions = [outputs["attentions"][i][:, -1] for i in layers]
                attentions = torch.stack(attentions, 1)
                # shape is [(batch_size, num_heads, input_length, input_length)]*num_layers

            hidden_states = torch.stack(
                [outputs["hidden_states"][i] for i in layers], 1
            )
            # (batch, layers, past_seq, logits) take just the last token so they are same size
            hidden_states = hidden_states[
                :, :, last_token
            ]  

        out = dict(
            hidden_states=hidden_states,
            attentions=attentions,
            scores=outputs["scores"],
            input_ids=input_ids,
        )
        out = {k: to_numpy(v) for k, v in out.items()}
        if debug:            
            out['input_truncated'] = self.tokenizer.batch_decode(input_ids)
            out['text_ans'] = self.tokenizer.batch_decode(outputs["scores"].argmax(-1))
            
        return out

    def get_layer_selection(self, outputs):
        """Sometimes we don't want to save all layers.

        Typically we can skip some to save space (stride). We might also want to ignore the first and last ones (padding) to avoid data leakage.

        See https://www.lesswrong.com/posts/bWxNPMy5MhPnQTzKz/what-discovering-latent-knowledge-did-and-did-not-find-4
        """
        return range(
            self.layer_padding,
            len(outputs["hidden_states"]) - self.layer_padding,
            self.layer_stride,
        )




    def batch_hidden_states(self, data: Dataset, n=100, batch_size=2, mcdropout=True):
        """
        Given an encoder-decoder model, a list of data, computes the contrast hidden states on n random examples.
        Returns numpy arrays of shape (n, hidden_dim) for each candidate label, along with a boolean numpy array of shape (n,)
        with the ground truth labels
        
        This is deliberately simple so that it's easy to understand, rather than being optimized for efficiency
        """
        
        ds_t_subset = data.select(range(n))
        ds_t_subset.set_format(type='torch', columns=['input_ids', 'label'])
        
        ds_p_subset = data.select(range(n))
        ds_p_subset.set_format(type="pandas", columns=['lie', 'label', 'prompt', 'prompt_truncated'])
        
        dl = DataLoader(ds_t_subset, batch_size=batch_size, shuffle=True)
        for i, batch in enumerate(tqdm(dl, desc='get hidden states')):
            input_ids, true_labels =  batch["input_ids"], batch["label"]
            nn = len(input_ids)
            index = i*batch_size+np.arange(nn)
            
            # different due to dropout
            hs1 = self.get_batch_of_hidden_states(input_ids=input_ids, use_mcdropout=mcdropout)
            if mcdropout:
                hs2 = self.get_batch_of_hidden_states(input_ids=input_ids, use_mcdropout=mcdropout)
                
                # QC
                if i==0:
                    eps=1e-5
                    mpe = lambda x,y: np.mean(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
                    a,b=hs2['hidden_states'],hs1['hidden_states']
                    assert mpe(a,b)>eps, "the hidden state pairs should be different but are not. Check model.config.use_cache==False, check this model has dropout in it's arch"
                    
                    # FIXME, move check to loading?
                    # assert ((hs1['prob_y']+hs1['prob_n'])>0.5).all(), "your chosen binary answers should take up a lot of the prob space, otherwise choose differen't tokens"
            else:
                hs2 = hs1

            
            for j in range(nn):
                # let's add the non torch metadata like label, prompt, lie, etc
                k = i*batch_size + j
                info = ds_p_subset[k]
                
                yield dict(
                    hs1=hs1['hidden_states'][j],
                    scores1=hs1["scores"][j],
                    
                    hs2=hs2['hidden_states'][j],
                    scores2=hs2["scores"][j],                    
                    
                    true=true_labels[j].item(),
                    index=index[j],
                    
                    **info
                )
                
                
        def __getstate__(self):
            """So avoid datasets trying to pickle a model lets set a custom pickle method"""
            state = self.__dict__.copy()
            state['model_config'] = self.model.config
            state['model_name'] = self.model.config
            del state['model']
            return state
    
        def __setstate__(self):
            raise NotImplementedError("You should not be pickling this class, it's too big")
