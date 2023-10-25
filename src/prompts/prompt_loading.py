"""
Modified from https://github.com/EleutherAI/elk/blob/3bbe26c3858aac1b03e6f80628a5056fae44db9c/elk/extraction/prompt_loading.py#L129

Changed to record choices
"""
from collections import Counter
from random import Random
from typing import Any, Iterator, Literal, List, Dict
from pathlib import Path
from datasets import ClassLabel, Dataset, Value, load_dataset
import yaml
import numpy as np
from elk.promptsource.templates import env
from elk.promptsource import DatasetTemplates
from elk.utils import (
    assert_type,
    infer_label_column,
    select_split,
)
import functools
from elk.extraction.balanced_sampler import BalancedSampler, FewShotSampler
import pandas as pd
from loguru import logger
from src.helpers.ds import shuffle_dataset_by
from src.datasets.scores import choice2id, choice2ids, row_choice_ids
from src.extraction.config import ExtractConfig
from src.models.load import verbose_change_param, AutoConfig, AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase


# Local path to the folder containing the templates
TEMPLATES_FOLDER_PATH = Path(__file__).parent / "templates"

def load_prompt_structure(path='structure.yaml', prompt_format='llama'):
    f = TEMPLATES_FOLDER_PATH / path
    yaml_dict = yaml.load(f.open('r'), Loader=yaml.FullLoader)
    templates = yaml_dict["templates"]
    jinja = templates[prompt_format]
    prompt_template = env.from_string(jinja)
    return prompt_template


def load_default_sys_instructions(path='system.yaml'):
    f = TEMPLATES_FOLDER_PATH / path
    yaml_dict = yaml.load(f.open('r'), Loader=yaml.FullLoader)
    templates = yaml_dict["templates"]["falsity"]
    return templates

default_sys_instructions = load_default_sys_instructions()


# @functools.lru_cache()
# def count_tokens(s):
#     return len(tokenizer(s).input_ids)

# def answer_len(answer_choices: list):
#     a = count_tokens(answer_choices[0])
#     b = count_tokens(answer_choices[1])
#     return max(a, b)

def sample_n_true_y_false_prompts(prompts, num_truth=1, num_lie=1, seed=42):
    """sample some truth and some false"""
    df = pd.DataFrame(prompts)
    
    # restrict to template where the choices are a single token
    # m = df.answer_choices.map(answer_len)<=2
    # df = df[m]
    df = pd.concat([
        df.query("instructed_to_lie==True").sample(num_truth, random_state=seed),
        df.query("instructed_to_lie==False").sample(num_lie, random_state=seed)])
    return df.to_dict(orient="records")

def load_prompts(
    ds_string: str,
    *,
    sys_instructions: Dict[bool, Dict[str, str]]= default_sys_instructions,
    binarize: bool = True,
    num_shots: int = 0,
    seed: int = 42,
    split_type: Literal["train", "val"] = "train",
    template_path: str | None = None,
    rank: int = 0,
    world_size: int = 1,
    prompt_format: str="chatml",
    prompt_sampler = sample_n_true_y_false_prompts,
    N=np.inf,
) -> Iterator[dict]:
    """Load a dataset full of prompts generated from the specified dataset.

    Args:
        ds_string: Name of HF dataset to use, e.g. `"super_glue:boolq"` or `"imdb"`.
        binarize: Whether to binarize the dataset labels for multi-class datasets.
        num_shots: The number of examples to use in few-shot prompts. If zero, prompts
            are zero-shot.
        seed: The seed to use for prompt randomization.
        split_type: Whether to use the train or val split of the dataset.
        template_path: Path to feed into `DatasetTemplates` for loading templates.
        rank: The rank of the current process. Defaults to 0.
        world_size: The number of processes. Defaults to 1.
        prompt_format: which prompt format to use e.g. vicuna, llama, chatml
        prompt_sampler: when given an unbalanced set of true and false prompts this might take one of each randomly

    Returns:
        An iterable of prompt dictionaries.
    """
    ds_name, _, config_name = ds_string.partition(":")

    ds_dict = assert_type(dict, load_dataset(ds_name, config_name or None))
    split_name = select_split(ds_dict, split_type)

    # TODO:, can I make sure it's the same shuffle regardless of length?
    ds = assert_type(Dataset, ds_dict[split_name].shuffle(seed=seed))
    if world_size > 1:
        ds = ds.shard(world_size, rank)

    if template_path is None:
        prompter = DatasetTemplates(ds_name, config_name)
    else:
        prompter = DatasetTemplates(template_path)

    # If the prompt template says to binarize, we should
    binarize = binarize or prompter.binarize
    prompter.drop_non_mc_templates()

    num_templates = len(prompter.templates)
    assert num_templates > 0
    if rank == 0:
        print(f"Extracting {num_templates} variants of each prompt")

    label_column = prompter.label_column or infer_label_column(ds.features)

    label_feature = ds.features[label_column]
    if isinstance(label_feature, ClassLabel):
        label_choices = [label_feature.str2int(label) for label in label_feature.names]
    elif isinstance(label_feature, Value) and label_feature.dtype == "bool":
        label_choices = [False, True]
    else:
        # Which classes are actually present in this split of the dataset?
        # This is shockingly fast since it uses an optimized Apache Arrow primitive.
        label_choices = sorted(ds.unique(label_column))
        if rank == 0:
            print(f"Using the following pseudo-labels: {label_choices}")

    rng = Random(seed)
    if num_shots > 0:
        train_name = select_split(ds_dict, "train")
        
        # TODO don't we need to binarize this?
        fewshot = FewShotSampler(
            ds_dict[train_name].shuffle(seed=seed),  # TODO: not iterator
            num_shots=num_shots,
            rng=rng,
            label_col=label_column,
        )
        fewshot_iter = iter(fewshot)
    else:
        fewshot_iter = None

    if label_column in ds.features:
        ds = BalancedSampler(
            ds.to_iterable_dataset(),
            set(label_choices),
            label_col=label_column,
        )
    else:
        if rank == 0:
            print("No label column found, not balancing")
        ds = ds.to_iterable_dataset()

    j = 0
    for i, example in enumerate(ds):
        if j>N:
            break
        prompts = _convert_to_prompts(
            example,
            binarize=binarize,
            label_column=label_column,
            label_choices=label_choices,  # type: ignore[arg-type]
            prompter=prompter,
            rng=rng,
            sys_instructions=sys_instructions,
            fewshot_iter=fewshot_iter,
            prompt_format=prompt_format,
        )
        prompts = [{'ds_string': ds_string, 'example_i':i, **p} for p in prompts]
        
        def prompt_ok(prompt):
            """ we want answers where we can distinguish them from the first token
            we don't have access to the tokenizer here, so we just make sure the first 3 letters are differen't and there are not spaces
            """
            answer_choices = prompt['answer_choices']
            a = answer_choices[0][:3]
            b = answer_choices[1][:3]
            keep = (a != b) and ' ' not in a
            if not keep:
                logger.debug(f"removing prompt because it's answers are not unique: {prompt['ds_string']} {prompt['template_name']} {prompt['answer_choices']}")
            return keep

        prompts = list(filter(prompt_ok, prompts))
        prompts = prompt_sampler(prompts, seed=42+j)
        # TODO: make sure they are single token answers (or at least the first token is unique)
        for p in prompts:
            j += 1
            yield p




def _convert_to_prompts(
    example: dict[str, Any],
    prompter: DatasetTemplates,
    binarize: bool,
    label_column: str,
    label_choices: list[bool | int | str],
    rng: Random,
    sys_instructions: Dict[bool, Dict[str, str]] = default_sys_instructions,
    fewshot_iter: Iterator[list[dict]] | None = None,
    prompt_format: str = "chatml",
) -> list:
    """Prompt-generating function to pass to `IterableDataset.map`."""
    prompt_template = load_prompt_structure(prompt_format=prompt_format)
    
    prompts = []
    templates = list(prompter.templates.values())

    # For sanity checking that prompts are unique
    prompt_counter = Counter()
    label = example[label_column]

    if binarize:
        # Replace the full list of possibilities with a randomly sampled false label
        # and the correct label, as done in the DLK paper. Note that this does add some
        # "supervision" by stacking the deck in favor of the correct answer.
        label_choices = [
            rng.choice([c for c in label_choices if c != label]),
            label,
        ]
        rng.shuffle(label_choices)

    for template in templates:
        answer_choices=template.get_fixed_answer_choices_list()
        
        # skip prompts where the responses are similar in the first token
        if answer_choices[0][:3]==answer_choices[1][:3]:
            logger.trace(f"skipping prompt because it's answers are not unique (for the first token): {template.name} {answer_choices}")
            continue
        answer_choices = [[c] for c in answer_choices]
        for instructed_to_lie in [False, True]:
            for sys_instr_name, sys_instr in sys_instructions[instructed_to_lie].items():
                fake_example = example.copy()
                if instructed_to_lie: fake_example['label'] = int(fake_example['label']==0)

                q, a = template.apply(fake_example)
                prompt_parts = [dict(user=q)]
                prompt_counter[(sys_instr + q, a)] += 1

                if fewshot_iter is not None:
                    # Infinite iterator so we don't need to worry about StopIteration
                    fewshot_examples = next(fewshot_iter)
                    if instructed_to_lie: 
                        fewshot_examples = [{**e, 'label': e['label']^0} for e in fewshot_examples]
                        for e in fewshot_examples:
                            # arg, check out negation worked
                            assert e['label']>=0
                            assert e['label']<2
                        
                    fewshot_texts = [
                        dict(user=q, response=a.strip()) for q, a in map(template.apply, fewshot_examples)
                    ]
                    for d in fewshot_texts:
                        # some of the answers have extra trailing text, that's OK. But extra preceeding text is not, let's check for that
                        assert any([any([d['response'].startswith(a) for a in ac]) for ac in answer_choices]), f"fewshot response `{d['response']}` has extra preceeding text compared to allowed choices: {answer_choices}. template is: {template.name}"
                    prompt_parts = fewshot_texts + prompt_parts
                    
                prompt_parts[0]['system'] = sys_instr
                
                q = "".join([prompt_template.render(**p) for p in prompt_parts])

                prompts.append(dict(
                    # Strip whitespace from the answer to make it easier to
                    # compare with the model's output
                    answer=a.strip(),
                    question=q,
                    
                    answer_choices=answer_choices,
                    template_name=template.name,
                    label_true=example['label'],
                    label_instructed=fake_example['label'],
                    instructed_to_lie=instructed_to_lie,
                    sys_instr_name=sys_instr_name,
                ))

    # Sanity check: variants should be unique
    ((maybe_dup, dup_count),) = prompt_counter.most_common(1)
    if dup_count > 1:
        raise ValueError(f'Prompt duplicated {dup_count} times! "{maybe_dup}"')

    return prompts



def load_preproc_dataset(ds_name: str, tokenizer: PreTrainedTokenizerBase, N:int, split_type:str="train", seed=42, num_shots=1, max_length=999) -> Dataset:
    """load a preprocessed dataset of tokens."""
    ds_prompts = Dataset.from_generator(
        load_prompts,
        gen_kwargs=dict(
            ds_string=ds_name,
            num_shots=num_shots,
            split_type=split_type,
            # template_path=template_path,
            seed=seed,
            prompt_format='llama',
            N=N*3,
        ),
    )

    # ## Format prompts
    # The prompt is the thing we most often have to change and debug. So we do it explicitly here.
    # We do it as transforms on a huggingface dataset.
    # In this case we use multishot examples from train, and use the test set to generated the hidden states dataset. We will test generalisation on a whole new dataset.

    ds_tokens = (
        ds_prompts
        .map(
            lambda ex: tokenizer(
                ex["question"], padding="max_length", max_length=max_length, truncation=True, add_special_tokens=True,
                return_tensors="pt",
                return_attention_mask=True,
                # return_overflowing_tokens=True,
            ),
            batched=True,
            desc='tokenize'
        )
        .map(lambda r: {"truncated": np.sum(r["attention_mask"], 0)==max_length}, desc='truncated')
        .map(
            lambda r: {"prompt_truncated": tokenizer.batch_decode(r["input_ids"])},
            batched=True,
            desc='prompt_truncated',
        )
        .map(lambda r: {'choice_ids': row_choice_ids(r, tokenizer)}, desc='choice_ids')
    )
    
    
    ds_tokens = shuffle_dataset_by(ds_tokens, 'example_i')
    print('num_rows', ds_tokens.num_rows)
    
    # ## Filter out truncated examples
    ds_tokens = ds_tokens.filter(lambda r: not r['truncated'])
    print('num_rows (after filtering out truncated rows)', ds_tokens.num_rows)
    assert len(ds_tokens), f'No examples left after filtering out truncated rows, try a longer max_length than {max_length}'
    return ds_tokens.select(range(N))
