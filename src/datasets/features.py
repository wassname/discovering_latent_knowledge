from datasets.features import Sequence, Value, Features, Array2D, Array3D, Features
from src.extraction.config import ExtractConfig
from transformers import AutoConfig

def get_features(cfg: ExtractConfig, config: AutoConfig) -> Features:
    
    
    l = config.num_hidden_layers+1
    h = config.hidden_size
    interventions = 2
    v = config.vocab_size

    features = {
        "end_hidden_states": Array3D(dtype="float32", id=None, shape=(l, h, interventions)),
        "end_logits": Array2D(dtype="float32", id=None, shape=(v, interventions)),
        "choice_probs": Array2D(dtype="float32", id=None, shape=(2, interventions)),
        "label_true": Value(dtype="bool", id=None),
        "instructed_to_lie": Value(dtype="bool", id=None),
        "question": Value(dtype="string", id=None),
        "answer_choices": Sequence(
            feature=Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            length=-1,
            id=None,
        ),
        "choice_ids": Sequence(
            feature=Sequence(feature=Value(dtype="int64", id=None), length=-1, id=None),
            length=-1,
            id=None,
        ),
        "template_name": Value(dtype="string", id=None),
        "sys_instr_name": Value(dtype="string", id=None),
        "example_i": Value(dtype="int64", id=None),
        "input_truncated": Value(dtype="string", id=None),
        "truncated": Value(dtype="bool", id=None),
        "text_ans": Value(dtype="string", id=None),
        "ans": Sequence(feature=Value(dtype="float32", id=None), length=-1, id=None),
    }
    return Features(features)
