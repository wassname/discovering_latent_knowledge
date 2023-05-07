```sh
conda create -n dlk2 python=3.9 -y
conda activate dlk2
mamba install -y pytorch torchvision torchaudio pytorch-cuda=11.7 cudatoolkit-dev==11.7  cudatoolkit=11.7 -c pytorch -c nvidia  -c conda-forge
mamba install -y ipykernel  pip
pip install -r requirements.txt
```
