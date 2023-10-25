from simple_parsing import Serializable, field
from dataclasses import InitVar, dataclass, replace
from typing import Literal

@dataclass
class ExtractConfig(Serializable):
    """Config for extracting hidden states from a language model."""

    datasets: tuple[str, ...] = ("amazon_polarity",  "super_glue:boolq", "glue:qnli", "imdb")
    """Names of HF datasets to use, e.g. `"super_glue:boolq"` or `"imdb"` `"glue:qnli"""
    
    model: str = "TheBloke/WizardCoder-Python-13B-V1.0-GPTQ"
    """HF model string identifying the language model to extract hidden states from."""

    data_dirs: tuple[str, ...] = ()
    """Directory to use for caching the hiddens. Defaults to `HF_DATASETS_CACHE`."""

    # int4: bool = True
    # """Whether to perform inference in mixed int8 precision with `bitsandbytes`."""

    max_examples: tuple[int, int] = (600, 600)
    """Maximum number of examples to use from each split of the dataset."""

    num_shots: int = 1
    """Number of examples for few-shot prompts. If zero, prompts are zero-shot."""

    num_variants: int = -1
    """The number of prompt templates to use for each example. If -1, all available
    templates are used."""

    layers: tuple[int, ...] = ()
    """Indices of layers to extract hidden states from. We follow the HF convention, so
    0 is the embedding, and 1 is the output of the first transformer layer."""

    layer_stride: InitVar[int] = 4
    """Shortcut for `layers = (0,) + tuple(range(1, num_layers + 1, stride))`."""
    
    layer_padding: InitVar[int] = 4
    """Skips this amount of first layers"""

    seed: int = 42
    """Seed to use for prompt randomization. Defaults to 42."""

    # token_loc: Literal["first", "last", "mean"] = "last"
    # """The location of the token to extract hidden states from."""
    
    template_path: str | None = None
    """Path to pass into `DatasetTemplates`. By default we use the dataset name."""
    
    max_length: int | None = 666
    """Maximum length of the input sequence passed to the tokenize encoder function"""
