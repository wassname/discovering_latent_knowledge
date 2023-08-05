
# LLM truth detector using Monte Carlo Dropout

Sometimes the best way to explain is with code:

```py
"""
pseudocode for a LLM truth detector using Monte Carlo Dropout
"""

# load model
model, tokenizer = load_model()

# make dataset of hidden state pairs
prompts = ["Is this true: a broken mirror gives 7 years bad luck [Yes/No]: ", 
           "Is this true: a broken mirror doesn't give 7 years bad luck [Yes/No]: "]
choice = ['Yes']
choice_is_true = [-1, 1]

def get_hidden_state_pairs(prompts, choice, choice_is_true, tokenizer, model):
  """
  We turn on dropout and predict the next token (repeat x2). Since dropout is turned on each prediction is slightly different.
  Then we collect the hidden state pairs (as x1, x2) and the scores of our target token (as y1, y2)
  """

  choice_tokens = choice2token(choice, tokenizer)

  # we enable dropout, and do 2 inferences that are slightly different
  enable_mcdropout(model)

  outputs1 = model.generate(prompts, output_hidden_states=True)
  y1 = outputs1['scores'][choice_tokens]
  x1 = outputs1["hidden_states"]

  outputs2 = model.generate(prompts, output_hidden_states=True)
  y2 = outputs2['scores'][choice_tokens]
  x2 = outputs2["hidden_states"]
  return x1, x2, y1, y2, choice_is_true

dataset = batched(get_hidden_state_pairs(prompts, choice, choice_is_true, model, tokenizer))

# now train a probe
net = Probe(layers=2, hs=32)
optim = Optim(lr=3e-4)
for x1, x2, y1, y2, choice_is_true in dl:
  y_pred1 = net(x1)
  y_pred2 = net(x2)
  y_pred = y_pred2-ypred1

  # our label is the distance between the two probabilities in the direction of truth
  # So if y2 is less true than y1, and they are 0.02% apart then y is -0.02%
  y = (y2-y1)*choice_is_true 

  # Use a MSE loss so that the distance between the predicted pair of scores (in the direction of truth)
  # is the same as the real pair of scores (in the direction of truth)
  loss = F.mse(y_pred, y)
  net.backwards()
  optim.step()

# Test the probe
prompts = ["Is this true: Ancients did not believe the world was flat [Yes/No]: ", 
           "Is this true: Step on a crack break your fathers back [Yes/No]: "]
choice = ['Yes']
choice_is_true = [1, -1]
x1, x2, y1, y2, choice_is_true = get_hidden_state_pairs(prompts, choice, choice_is_true, model, tokenizer)
y_pred1 = net(x1)
y_pred2 = net(x2)

# translate this into a truth detector....
pred_last_choice_is_true = y / (y_pred2-y_pred)
pred_last_choice_is_true # [1, -1]
```

# Description
There is some previous work on this ([ELK](https://github.com/EleutherAI/elk), [DLK](https://github.com/collin-burns/discovering_latent_knowledge/blob/main/CCS.ipynb), CSS, etc) that all take varias approaches. They have this in common:

- Show the model 2 statements “the sky is blue” “the sky is green”
- Get the hidden states from reading those statements
- Use machine learning learning to distinguish between those two sets

Now this works well [(or not?)](https://www.lesswrong.com/posts/bWxNPMy5MhPnQTzKz/what-discovering-latent-knowledge-did-and-did-not-find-4), but I aim for two improvements:
- Detect direction of deception instead of truth
- look at deceptive actions (outputs), not deceptive observations (inputs).
- Use Monte Carlo dropout to generate pair of hidden states, instead pairs of inputs


My contributions/finds so far:
- Instead of comparing hidden states from 2 prompts, you can compare two inferences of the same prompt as long as you have dropout on
  - For this pair of hidden states, one will be in the direction of truth and one will not
    - But the pairs must give >10% differen't answer on our compared tokens e.g. true vs false
  - We can detect this using a supervised probe (with 90% acc on IMBD sentiment analysis)
- The best approach to setting up the probe is ~~binary classification~~, ~~multiclass classification~~ ~~ranking with margin_ranking_loss~~ ranking with L1smoothloss
  - This is because treating it like a ranking problem decreases overfitting
  - And learning distance and direction between the ranked pairs gives more supervision than just the direction (like in many ranking setups)
- It's hard to get models to lie! Even for uncensored models. I find uncensored coding models are best


## TODO:

I'm trying to 

- [x] use pytorch lightning
- [x] batch hidden states 5x faster
- [x] use wizcoer 15B, to see if larger models give better results
- [x] eval on some deceptive or misleading statements
- [x] debug by looking at model output
- [x] test generalization
- [x] try differen't approaches
  - [x] setup
    - [x] detect deception vs truth
    - [x] differen't prompts
    - [x] differen't tasks
  - [x] model arch
    - [x] put in both states
    - [x] normalize states
    - [x] mix states at end

-------------

# Discovering Latent Knowledge Without Supervision

This repository contains the essential code for Discovering Latent Knowledge in Language Models Without Supervision.

<p align="center">
<img src="figure.png" width="750">
</p>

We introduce a method for discovering truth-like features directly from model activations in a purely unsupervised way.

## Abstract
> Existing techniques for training language models can be misaligned with the truth: if we train models with imitation learning, they may reproduce errors that humans make; if we train them to generate text that humans rate highly, they may output errors that human evaluators can't detect. We propose circumventing this issue by directly finding latent knowledge inside the internal activations of a language model in a purely unsupervised way. Specifically, we introduce a method for accurately answering yes-no questions given only unlabeled model activations. It works by finding a direction in activation space that satisfies logical consistency properties, such as that a statement and its negation have opposite truth values. We show that despite using no supervision and no model outputs, our method can recover diverse knowledge represented in large language models: across 6 models and 10 question-answering datasets, it outperforms zero-shot accuracy by 4\% on average. We also find that it cuts prompt sensitivity in half and continues to maintain high accuracy even when models are prompted to generate incorrect answers. Our results provide an initial step toward discovering what language models know, distinct from what they say, even when we don't have access to explicit ground truth labels.

## Code

We provide three options for code:
1. A notebook walking through our main method in a simple way: `CCS.ipynb`. This may be the best place to start if you want to understand the method better and play around with it a bit.
2. More flexible and efficient scripts for using our method in different settings: `generate.py` and `evaluate.py` (both of which rely heavily on `utils.py`). This code is a polished and simplified version of the code used for the paper. This may be the best place to build on if you want to build on our work.
3. You can also download the original (more comprehensive, but also more complicated and less polished) code [here](https://openreview.net/attachment?id=ETKGuby0hcs&name=supplementary_material).

Below we provide usage details for our main python scripts (`generate.py` and `evaluate.py`).

### Generation
First, use `generate.py` for (1) creating contrast pairs, and (2) generating hidden states from a model. For example, you can run:
```
python generate.py --model_name deberta  --num_examples 400 --batch_size 40
```
or
```
python generate.py --model_name gpt-j --num_examples 100 --batch_size 20
```
or
```
CUDA_VISIBLE_DEVICES=0,1 python generate.py --parallelize --model_name t5-11b --num_examples 100 
```

To use the decoder of an encoder-decoder model (which we found is worse than the encoder for T5 and UnifiedQA, but better than the encoder for T0), specify `--use_decoder`.

There are also many optional flags for specifying the dataset (`--dataset`; the default is `imdb`), the cache directory for model weights (`--cache_dir`; the default is `None`), which prompt for the dataset to use (`--prompt_idx`; the default is `0`), where to save the hidden states for all layers in the model (`--all_layers`), and so on.

### Evaluation 
After generating hidden states, you can use `evaluate.py` for running our main method, CCS, on those hidden states. Simply run it with the same flags as you used when running `generate.py`, and it will load the correct hidden states for you. For example, if you ran `generate.py` with DeBERTa, generating 400 examples with a batch size of 40 (the first example in the Generation section), then you can run:
```
python evaluate.py --model_name deberta  --num_examples 400 --batch_size 40
```

In addition to evaluating the performance of CCS, `evaluate.py` also verifies that logistic regression (LR) accuracy is reasonable. This can diagnose why CCS performance may be low; if LR accuracy is low, that suggestions that the model's representations aren't good enough for CCS to work well.

### Requirements

This code base was tested on Python 3.7.5 and PyTorch 1.12. It also uses the [datasets](https://pypi.org/project/datasets/) and [promptsource](https://github.com/bigscience-workshop/promptsource) packages for loading and formatting datasets. 

## Citation

If you find this work helpful, please consider citing our paper:

    @article{burns2022dl,
      title={Discovering Latent Knowledge in Language Models Without Supervision},
      author={Burns, Collin and Ye, Haotian and Klein, Dan and Steinhardt, Jacob},
      journal={ArXiV},
      year={2022}
    }

