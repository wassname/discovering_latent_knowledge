"""
Some tools modified from honest_llama.

https://github.com/likenneth/honest_llama/blob/master/utils.py#L645
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Union, NewType
from einops import rearrange, reduce, repeat, asnumpy, parse_shape
import torch
import pickle
from src.config import root_folder
from src.prompts.prompt_loading import load_preproc_dataset
from transformers import AutoTokenizer, pipeline, Pipeline
from loguru import logger
from jaxtyping import Float
from torch import nn, Tensor


def create_cache_interventions(
    model,
    tokenizer,
    cfg,
    # N_fit_examples=20,
    # batch_size=2,
    rep_token=-1,
    n_difference=1,
    # direction_method="pca",
    get_negative=False,
):
    """
    We want one set of interventions per model

    So we always load a cached version if possible. to make it approx repeatable use the same dataset etc
    """
    direction_method = cfg.intervention_direction_method
    N_fit_examples = cfg.intervention_fit_examples
    batch_size = cfg.batch_size
    tokenizer_args = dict(
        padding="max_length",
        max_length=cfg.max_length,
        truncation=True,
        add_special_tokens=True,
    )

    hidden_layers = list(
        range(cfg.layer_padding, model.config.num_hidden_layers, cfg.layer_stride)
    )
    ll = sum(hidden_layers)
    model_name = cfg.model.replace("/", "-")
    intervention_f = (
        root_folder
        / "data"
        / "interventions"
        / f'{model_name}_{"-" if get_negative else "+"}_{direction_method}_{ll}.pkl'
    )
    intervention_f.parent.mkdir(exist_ok=True, parents=True)
    if not intervention_f.exists():
        dataset_fit = load_preproc_dataset(
            "imdb",
            tokenizer,
            N=N_fit_examples,
            seed=cfg.seed,
            num_shots=cfg.num_shots,
            max_length=cfg.max_length,
            prompt_format=cfg.prompt_format,
        )

        train_labels = np.array(dataset_fit["label_true"])
        if get_negative:
            assert direction_method not in ["pca"], "PCA does not have a direction"
            assert isinstance(train_labels[0], bool), "train_labels must be bool"
            train_labels = train_labels == 0

        rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)
        intervention = rep_reading_pipeline.get_directions(
            dataset_fit["question"],
            rep_token=rep_token,
            hidden_layers=hidden_layers,
            n_difference=n_difference,
            train_labels=dataset_fit["label_true"],
            direction_method=direction_method,
            batch_size=batch_size,
            layer_name_tmpl=cfg.intervention_layer_name_template,
            **tokenizer_args,
        )

        assert np.isfinite(np.concatenate(list(intervention.direction.values()))).all()
        # assert torch.isfinite(torch.concat(list(honesty_rep_reader.directions.values()))).all()
        # and save
        with open(intervention_f, "wb") as f:
            pickle.dump(intervention, f)
            logger.info(f"Saved interventions to {intervention_f}")

    with open(intervention_f, "rb") as f:
        intervention = pickle.load(f)
    logger.info(f"Loaded interventions from {intervention_f}")

    return intervention


def test_intervention_quality(
    dataset_train, intervention, model, rep_control_pipeline2, batch_size=2, ds_name=""
):
    """
    Check interventions are ordered and different and valid
    """
    # TODO over multiple batches?
    inputs = dataset_train  # [:batch_size]
    model.eval()
    baseline_outputs = []
    for batch_index in range(len(inputs) // batch_size):
        batch = inputs[batch_index * batch_size : (batch_index + 1) * batch_size]
        with torch.no_grad():
            baseline_outputs += rep_control_pipeline2(
                batch, batch_size=batch_size, intervention=intervention
            )

    # So here we check that the interventions are ordered, e.g. the positive one gives a more true answer than the neutral or negative ones
    print(
        f"Testing intervention quality on {ds_name}. results should be different and usually ordered"
    )
    data = []
    coverage = []
    for bi in range(len(baseline_outputs)):
        r = baseline_outputs[bi]

        # residuals = r['end_hidden_states'].diff(0)
        choices = r["answer_choices"]
        label = r["label_true"]
        ans = r["ans"].numpy()

        # TODO coverage too
        # mean_prob = np.sum(r['choice_probs'], 1).mean()
        coverage.append(torch.sum(r["choice_probs"], 1))

        choice_true = choices[label]
        if label == 0:
            ans = 1 - ans  # reverse it

        ordered = (np.argsort(ans) == np.arange(len(ans))).all()
        data.append(
            dict(
                order=ordered,
                diff=(abs(np.diff(ans)) > 0.1).item(),
                adj_ans=ans,
                label=label,
            )
        )
        if bi < 5:
            print(f"\t Score: {list(ans)} of true ans `{choice_true}`=={label}")
            print(f"\t Ordered? {ordered} {np.argsort(ans)}")
            print(f"\t Different? {abs(np.diff(ans))>0.1} {np.diff(ans)}")
            print("'\t top choices", r["text_ans"], "should be valid")
    df = pd.DataFrame(data)
    print(f"N={len(df)}")
    print(f"rows that are ordered {df['order'].mean():%}")
    print(f"rows that are different {df['diff'].mean():%}")

    # note if our intervention is too big we will output garbage, and our valid choices will be low prob - this is a sign of a poor intervention
    coverage = torch.stack(coverage)
    print(f"mean choice coverage by intervention {coverage.mean(0)}")
    return df


from IPython.display import display, HTML


def top_toke_probs(
    o,
    tokenizer,
    N=20,
):
    """
        nicely return top token probabilities e.g.

            prob_0	tokens_0	id_0	prob_1	tokens_1	id_1
    0	0.811495	`Neg`	32863	0.666955	`Neg`	32863
    1	0.113310	`Pos`	21604	0.194095	`Pos`	21604
    2	0.020635	`Ne`	8199	0.065014	`Ne`	8199
    """
    data = {}
    for i in range(o["end_logits"].shape[1]):
        probs = torch.softmax(o["end_logits"][:, i], -1)
        top = probs.argsort(0, descending=True)
        top_probs = probs[top]
        tokens_top20 = tokenizer.batch_decode(
            top[:N], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        tokens_top20 = [f"`{t}`" for t in tokens_top20]
        data.update(
            {
                f"prob_{i}": top_probs[:N],
                f"tokens_{i}": tokens_top20,
                f"id_{i}": top[:N],
            }
        )
    return pd.DataFrame(data)


def print_pipeline_row(o: dict, tokenizer):
    """take in single pipeline output, and prince intervention metrics."""
    choices = [tokenizer.batch_decode(cc) for cc in o["choice_ids"]]
    index = [o[0] for o in choices]
    d = pd.DataFrame(
        o["choice_probs"].numpy(), columns=["edit=None", "edit=+"], index=index
    ).T
    # d["top1 coverage"] = d.sum(1)
    mean_prob = o["choice_probs"].sum(0)
    d["coverage"] = mean_prob

    print("choices", choices)
    max_prob, max_token = torch.softmax(o["end_logits"][:, :], 0).max(0)
    max_detoken = tokenizer.batch_decode(max_token)
    d["top_token"] = max_detoken
    d["top_prob"] = max_prob
    d["label_true"] = o["label_true"]
    d["label_instructed"] = o["label_instructed"]
    print("choice probs")
    display(d)

    d1 = top_toke_probs(o, tokenizer)
    print("top token probs")
    display(d1)
