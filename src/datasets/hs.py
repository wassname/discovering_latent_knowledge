from dataclasses import dataclass
import lightning as pl
import torch
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
from typing import Optional, List, Tuple
from transformers import LogitsProcessorList

from src.helpers.torch import to_numpy
from src.datasets.dropout import enable_dropout


def get_choices_as_tokens(
    tokenizer, choice_n: List[str] = ["Negative"], choice_p: List[str] = ["Positive"]
) -> Tuple[List[int], List[int]]:
    # Note some tokenizer differentiate between "no", "\nno", so we sometime need to add whitespace beforehand...
    ids_n = []
    for c in choice_n:
        id_ = tokenizer(f"\n{c}", add_special_tokens=True)["input_ids"][-1]
        ids_n.append(id_)
        assert tokenizer.decode([id_]) == c

    ids_y = []
    for c in choice_n:
        id_ = tokenizer(f"\n{c}", add_special_tokens=True)["input_ids"][-1]
        ids_y.append(id_)
        assert tokenizer.decode([id_]) == c

    return ids_n, ids_y


@dataclass
class ExtractHiddenStates:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    layer_stride: int = 1
    layer_padding: int = 2
    truncation_length = 999
    choices_n: List[str] = ["No"]
    choices_p: List[str] = ["Yes"]

    def start(self):
        self.ids_n, self.ids_y = get_choices_as_tokens(
            self.tokenizer, self.choices_n, self.choices_p
        )

    def get_hidden_states(
        self,
        input_text,
        truncation_length=999,
        output_attentions=False,
        use_mcdropout=True,
    ):
        """
        Given a decoder model and some texts, gets the hidden states (in a given layer) on that input texts
        """
        if not isinstance(input_text, list):
            input_text = [input_text]
        input_ids = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        ).input_ids.to(self.model.device)

        # Handling truncation: truncate start, not end
        if truncation_length is not None:
            if input_ids.size(1) > truncation_length:
                print("truncating", input_ids.size(1))
            input_ids = input_ids[:, -truncation_length:]

        # forward pass
        last_token = -1
        first_token = 0
        with torch.no_grad():
            self.model.eval()
            if use_mcdropout:
                enable_dropout(self.model, use_mcdropout)

            # taken from greedy_decode https://github.com/huggingface/transformers/blob/ba695c1efd55091e394eb59c90fb33ac3f9f0d41/src/transformers/generation/utils.py
            logits_processor = LogitsProcessorList()
            model_kwargs = dict(use_cache=False)
            model_inputs = self.model.prepare_inputs_for_generation(
                input_ids, **model_kwargs
            )
            outputs = self.model.forward(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=True,
            )

            next_token_logits = outputs.logits[:, last_token, :]
            outputs["scores"] = logits_processor(input_ids, next_token_logits)[
                :, None, :
            ]

            next_tokens = torch.argmax(outputs["scores"], dim=-1)
            outputs["sequences"] = torch.cat([input_ids, next_tokens], dim=-1)

            # the output is large, so we will just select what we want 1) the first token with[:, 0]
            # 2) selected layers with [layers]
            layers = self.get_layer_selection(outputs)
            attentions = None
            layers = range(
                self.layer_padding,
                len(outputs["attentions"]) - self.layer_padding,
                self.layer_stride,
            )
            if output_attentions:
                # shape is [(batch_size, num_heads, sequence_length, sequence_length)]*num_layers
                # lets take max?
                attentions = [outputs["attentions"][i] for i in layers]
                attentions = [v[:, last_token] for v in attentions]
                attentions = torch.concat(attentions)

            hidden_states = torch.stack(
                [outputs["hidden_states"][i] for i in layers], 1
            )

            hidden_states = hidden_states[
                :, :, last_token
            ]  # (batch, layers, past_seq, logits) take just the last token so they are same size

            input_truncated = self.tokenizer.batch_decode(input_ids)

            s = outputs["sequences"]
            s = [s[i][len(input_ids[i]) :] for i in range(len(s))]
            text_ans = self.tokenizer.batch_decode(s)

            scores = outputs["scores"][:, first_token].softmax(
                -1
            )  # for first (and only) token
            # prob_n, prob_y = scores[:, [id_n, id_y]].T
            prob_n = scores[:, self.ids_n]
            prob_y = scores[:, self.ids_y]
            eps = 1e-3
            ans = (prob_y / (prob_n + prob_y + eps)).sum(1)

        out = dict(
            hidden_states=hidden_states,
            ans=ans,
            text_ans=text_ans,
            input_truncated=input_truncated,
            input_id_shape=input_ids.shape,
            attentions=attentions,
            prob_n=prob_n,
            prob_y=prob_y,
            scores=outputs["scores"][:, 0],
            input_text=input_text,
        )
        out = {k: to_numpy(v) for k, v in out.items()}
        return out

    def get_layer_selection(self, outputs):
        """Sometimes we don't want to save all layers.

        Typically we can skip some to save space (stride). We might also want to ignore the first and last ones (padding) to avoid data leakage.

        See https://www.lesswrong.com/posts/bWxNPMy5MhPnQTzKz/what-discovering-latent-knowledge-did-and-did-not-find-4
        """
        return range(
            self.layer_padding,
            len(outputs["attentions"]) - self.layer_padding,
            self.layer_stride,
        )



