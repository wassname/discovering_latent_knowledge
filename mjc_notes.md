```sh
# note 
conda create -n dlk2 python=3.9 -y
conda activate dlk2
mamba install -y pytorch torchvision torchaudio pytorch-cuda=11.7 cudatoolkit-dev==11.7  cudatoolkit=11.7 -c pytorch -c nvidia  -c conda-forge
mamba install -y ipykernel pip
pip install -r requirements.txt
```

# 2023-05-13 15:17:05

- [x] Convert it to lightning
- [ ] batch for get hidden states
  - [x] and cache
  - [x] 9s vs 60. so 10x faster


# 2023-05-21 11:26:20

- [ ] BUG: for some reason the model it not working as zero shot
  - OK I don't think it's the prompt? it must be my tokens? Lets make a scratch notebook to try and just load llama correctly
- [ ] also I would like to eval on some custom deceptive statements

# 2023-05-25 17:07:37

Hmm is sigmoid a good idea?

Lets just get hideen states once

- [ ] make the answer good? maybe with one shot
- [ ] fix prompt. so we have qeustion, true answer


Start again:
- [ ] the model can deceive (test) or not (train)
- [ ] the model can give an answer both in generate and forward
- [x] we can get hidden states (cached)


What is each step actually doing?
- we are finding a latent space direction that correspond to truth, sure
- but truth of what? what it's read, or what it's generating!?!
- what's it read presumably. but that's not what we are interested in. We are interested in the truth of what it generated which is quite different
- so how to we get the truth of what it generated?


How about human? well you give me an article, and ask me to complete is. I notice a lie in the article? I decide to lie to you as well?.

Now you give me extended article to another ai, and ask if there is a lie in there? that might work
but we would rather know if the generated thing is true, from the mind of the one who generated it

so that means we actually need the hidden states DURING GENERATION!

now generation is slow. so we should make it generate just a y/n.


# 2023-05-28 10:12:54

I need a model that will lie to me for the test set?... they are not very consistent. 

Maybe with
- better search
- manual pruning of generations?

I guess this shows they they trained whether the text it read is true... because that's much simpler


# 2023-05-28 16:46:38

bug: so there are two no tokens... wtf
the model only uses one! wtf!

oh it's _No vs No. One is the start of a new word. E.g. " No" and "\nNo" are _No. But "No" is "No"


Q for elk
- why int16 quant of hidden states? oh for the datasets package! I see https://github.com/EleutherAI/elk/issues/208
- why use forward not generate?
  - forward is what it thought of the last token?
  - generate is what it thinks of the generated tokens, conditional on what it read. 
- The later seems much closer to mind reading, and much closer to what we want
- additionally I've made this change in a PR and get X results

# 2023-05-29 07:09:38

Note we are using normalized for sklearn and it seems easy?
but unnorm for CCS, hmm

I can't get it to reliably lie, even at 30B. Grrr. Maybe an unaligned model?

OK so I asked on discord
- why not use generate? A: too hard, and how to make sure it's an answer (solution is to use relative probs and a single work). So they just haven't gone there.
- and how to make it lie. one person used yes momentum, nad it worked for them.

I would like to try:
- [ ] lying larry. with a larry prompt. and larry response.
- [ ] I could also keep sampling untill I get the prob diff I want! :)
  - the second actually seems better. since it can make sure that the pairs are the same except the answer!!


so wait
- how often are the pairs opposite answers? :(

So with the tuples:
- sometimes both are the same, making one a lie
- sometimes they are contrasting

So I could collect...
- valid contrasting answer? ... this seem inelegant as I've got no garuntees how long it will take
- find some way to generate random inference hidden states... so far it seem deterministic... maybe use mc dropout!!


OK I'm trying to find a model with dropout... most of them dont
- llama no
- opeansssistant no
- redpyjamas? no
- falcan? has dropout but no effect
- "stabilityai/stablelm-tuned-alpha-7b" has dropout but no effect
- [pythia](https://huggingface.co/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5). has dropout but no effect
- dolly: has dropout but no effect
- 
**but it looks like most of them are usingt an attention path that bypasses**

worst case I can use that momentum. I'm still judging it on it's answer. It's just that I really want it to know it's lying.

side note: popular large models like LLaMA, Gopher, Chinchilla, GPT-3, and PaLM did not use dropout, since it can slow down learning.

- "stabilityai/stablelm-tuned-alpha-7b" # has dropout

# 2023-06-08 07:20:48

Goal: get it lie, knowing it can lie
Extra: get contrastive pairs

TODO:
- try monte carlo dropout?
  - get stablm working
    - need proper prompting?
    - then add dropout, and see if that helps with pairs..
    - (or maybe add noise? augmentation?)
- [ ] use talk to llm model cards
- [ ] just use forward...
- [ ] get some stats on yes, no, lie yes, lie no...
- [ ] find a way to randomize

How does generation work anyway? [so](https://github.com/huggingface/transformers/blob/ba695c1efd55091e394eb59c90fb33ac3f9f0d41/src/transformers/generation/utils.py#L2338) it looks like it's just a forward. where `next_token_logits = outputs.logits[:, -1, :]`. Where logits are from `lm_logits = self.lm_head(hidden_states)`

Note default logits_processor is LogitsProcessorList([]) where it doesn't do anything as it's empty


So how do a I turn a forward into the next token....
- I go `outputs.logits[:, -1, :].softmax(-1).argmax(-1)`


```
model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
outputs = model(**model_inputs, return_dict=True)
next_token_logits = outputs.logits[:, -1, :]
next_tokens_scores = logits_processor(input_ids, next_token_logits)
next_tokens = torch.argmax(next_tokens_scores, dim=-1)
```

OK so know I know that greedy 1 token generation IS the same as forward. BUT I still have the same problem. Am I seperating the model acting on a lie, or **knowingly generating a lie with high prob?**

:bug: why does mcdropout not work!?! Why is it deterministic? Can I inject noise?


Hmm base models seem better, since they are not trained for honesty!



So maybe I should see IF I can get many shot, acc=0.9 without lies. Then I can add lies.


stylaised knowledge:
- the prompt matters
- I don't know if the size of the model matters
- I don't know if the type of model matters


How to get the prompt? more direct. Just a lying one. Just a true one.
