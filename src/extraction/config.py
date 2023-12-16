from simple_parsing import Serializable, field
from dataclasses import InitVar, dataclass, replace
from typing import Literal

@dataclass
class ExtractConfig(Serializable):
    """Config for extracting hidden states from a language model."""

    datasets: tuple[str, ...] = ("amazon_polarity",  "super_glue:boolq", "glue:qnli", "imdb")
    """Names of HF datasets to use, e.g. `"super_glue:boolq"` or `"imdb"` `"glue:qnli"""
    
    # model: str = "TheBloke/WizardCoder-Python-13B-V1.0-GPTQ"
    # model: str = "TheBloke/Wizard-Vicuna-13B-Uncensored-GPTQ"
    # model: str = "TheBloke/Wizard-Vicuna-7B-Uncensored-GPTQ"
    # model: str = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ" # it wont lie? wtf
    # model: str = "microsoft/phi-2"
    # model: str = "microsoft/phi-2"
    # model: str = "/media/wassname/SGIronWolf/projects5/elk/phi-2"
    # model: str = "TheBloke/phi-2-GPTQ"
    model: str = "wassname/phi-2-GPTQ_w_hidden_states"
    
    # model: str = "TheBloke/Llama-2-13B-chat-GPTQ"
    """HF model string identifying the language model to extract hidden states from."""

    batch_size: int = 5

    pad_token_id: int = 50256
    """Token ID to use for padding, most often zero."""

    prompt_format: str | None = 'phi'
    """if the tokenizer does not have a chat template you can set a custom one. see src/prompts/templates/prompt_formats/readme.md."""
    
    data_dirs: tuple[str, ...] = ()
    """Directory to use for caching the hiddens. Defaults to `HF_DATASETS_CACHE`."""

    max_examples: tuple[int, int] = (1000, 200)
    """Maximum number of examples to use from each split of the dataset."""
    

    num_shots: int = 2
    """Number of examples for few-shot prompts. If zero, prompts are zero-shot."""

    num_variants: int = -1
    """The number of prompt templates to use for each example. If -1, all available
    templates are used."""

    layer_stride: InitVar[int] = 1
    """Shortcut for `layers = (0,) + tuple(range(1, num_layers + 1, stride))`."""
    
    layer_padding: InitVar[int] = 6
    """Skips this amount of first layers"""

    seed: int = 42
    """Seed to use for prompt randomization. Defaults to 42."""

    # token_loc: Literal["first", "last", "mean"] = "last"
    # """The location of the token to extract hidden states from."""
    
    template_path: str | None = None
    """Path to pass into `DatasetTemplates`. By default we use the dataset name."""
    
    max_length: int | None = 1000
    """Maximum length of the input sequence passed to the tokenize encoder function"""
    
    disable_ds_cache: bool = False
    """Disable huggingface datasets cache."""

    intervention_direction_method: str = "cluster_mean"
    """"how to intervent: pca, cluster_mean, random"""

    intervention_fit_examples: int = 200
    """how many example to use for intervention calibration"""

    intervention_layer_name_template: str = "transformer.h.{}"
    """path to model layers"""


