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
