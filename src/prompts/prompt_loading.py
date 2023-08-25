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
from elk.promptsource.templates import env
from elk.promptsource import DatasetTemplates
from elk.utils import (
    assert_type,
    infer_label_column,
    select_split,
)
from elk.extraction.balanced_sampler import BalancedSampler, FewShotSampler

# Local path to the folder containing the templates
TEMPLATES_FOLDER_PATH = Path(__file__).parent / "templates"

def load_prompt_structure(path='structure.yaml', prompt_format='chatml'):
    f = TEMPLATES_FOLDER_PATH / path
    yaml_dict = yaml.load(f.open('r'), Loader=yaml.FullLoader)
    templates = yaml_dict["templates"]
    jinja = templates[prompt_format]
    prompt_template = env.from_string(jinja)
    return prompt_template


def load_default_sys_instructions(path='system.yaml'):
    f = TEMPLATES_FOLDER_PATH / path
    yaml_dict = yaml.load(f.open('r'), Loader=yaml.FullLoader)
    templates = yaml_dict["templates"]
    return templates

default_sys_instructions = load_default_sys_instructions()


def load_prompts(
    ds_string: str,
    *,
    sys_instructions: Dict[bool, Dict[str, str]]= default_sys_instructions,
    binarize: bool = False,
    num_shots: int = 0,
    seed: int = 42,
    split_type: Literal["train", "val"] = "train",
    template_path: str | None = None,
    rank: int = 0,
    world_size: int = 1,
    prompt_format: str="chatml",
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

    Returns:
        An iterable of prompt dictionaries.
    """
    ds_name, _, config_name = ds_string.partition(":")

    ds_dict = assert_type(dict, load_dataset(ds_name, config_name or None))
    split_name = select_split(ds_dict, split_type)

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
        fewshot = FewShotSampler(
            ds_dict[train_name].shuffle(seed=seed),  # TODO: not iterator
            num_shots=num_shots,
            rng=rng,
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

    for example in ds:
        yield _convert_to_prompts(
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
) -> dict[str, Any]:
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
        for lie in [False, True]:
            for sys_instr_name, sys_instr in sys_instructions[lie].items():
                fake_example = example.copy()
                if lie: fake_example['label'] = int(fake_example['label']==0)

                q, a = template.apply(fake_example)
                prompt_parts = [dict(user=q)]
                prompt_counter[(sys_instr + q, a)] += 1

                if fewshot_iter is not None:
                    # Infinite iterator so we don't need to worry about StopIteration
                    fewshot_examples = next(fewshot_iter)
                    if lie: fewshot_examples = [{**e, 'label': ~e['label']} for e in fewshot_examples]
                    fewshot_texts = [
                        dict(user=q, response=a) for q, a in map(template.apply, fewshot_examples)
                    ]
                    prompt_parts = fewshot_texts + prompt_parts
                    
                prompt_parts[0]['system'] = sys_instr
                
                q = "".join([prompt_template.render(**p) for p in prompt_parts])

                prompts.append(dict(
                    # Strip whitespace from the answer to make it easier to
                    # compare with the model's output
                    answer=a.strip(),
                    question=q,
                    
                    answer_choices=template.get_fixed_answer_choices_list(),
                    template_name=template.name,
                    label_true=example['label'],
                    label_instructed=fake_example['label'],
                    instructed_to_lie=lie,
                    sys_instr_name=sys_instr_name,
                ))

    # Sanity check: variants should be unique
    ((maybe_dup, dup_count),) = prompt_counter.most_common(1)
    if dup_count > 1:
        raise ValueError(f'Prompt duplicated {dup_count} times! "{maybe_dup}"')

    # Our reporter training and evaluation code assumes that the labels are integers.
    # If they're not, we need to convert them with index(). label_choices is guaranteed
    # to be sorted (see above).
    return dict(
        label=label_choices.index(label),
        prompts=prompts,
        template_names=[template.name for template in templates],
    )
