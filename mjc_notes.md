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
