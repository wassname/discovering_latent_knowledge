import torch
import torch.nn as nn
import numpy as np
import lightning as pl
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from src.datasets.load import ds2df
from datasets.arrow_dataset import Dataset
from einops import rearrange, reduce, repeat
from src.helpers.ds import shuffle_dataset_by
from src.helpers import bool2switch, switch2bool


# def compute_distance(df):
#     """distance between ans1 and ans2."""
#     true_switch_sign = df.label_true*2-1 # switch sign to desired answer. with this we ask which is more true
#     # otherwise we ask which is more positive
#     distance = (df.ans1-df.ans0) * true_switch_sign
#     return distance

to_tensor = lambda x: x # torch.from_numpy(x).float()
to_ds = lambda hs0, hs1, y: TensorDataset(to_tensor(hs0), to_tensor(hs1), to_tensor(y))




class imdbHSDataModule(pl.LightningDataModule):
    

    def __init__(self,
                 ds: Dataset,
                 batch_size: int=32,
                 x_cols = ['end_hidden_states'],
                 skip_layers = 0,
                ):
        super().__init__()
        self.save_hyperparameters(ignore=["ds"])
        self.ds = ds
        self.x_cols = x_cols
        self.skip_layers = skip_layers

    def setup(self, stage: str):
        h = self.hparams
        
        # extract data set into N-Dim tensors and 1-d dataframe
        self.ds_hs = (
            self.ds.select_columns(self.x_cols)
        )
        df = self.df = ds2df(self.ds)
        switch = bool2switch(df['label_true']).values
        
        # probs_c = self.ds['ans']
        self.ans = self.ds['ans'] #probs_c[:, 1] / (np.sum(probs_c, 1) + 1e-5)
        # df['y'] = df['label_true'].values[:, None] == (self.ans > 0.5)       
        
        
        # take the llm prob towards the positive answer, and flip it if negative was true
        # giving us the llm's assigned prob toward the true answer
        self.prob_on_truth = torch.tensor(switch[:, None] * self.ans)
        
        b = len(self.ds_hs)
        # take the diff between layers. Shape batch, layers, hidden_states, inferences
        hs = torch.tensor(self.ds_hs['end_hidden_states'])
        hs = hs.diff(1, axis=1) # this makes it the residual between layers
        if self.skip_layers:
            hs = hs[:, self.skip_layers:] # drop the first 10 layers to prevent overfitting?
        self.hs0 = hs[..., 0]
        self.hs1 = hs[..., 1]
        
        # so we are trying to predict is one hidden state is more true than the other
        # or specifically the distance and direction on the truth axis
        self.y = self.prob_on_truth[:, 1] - self.prob_on_truth[:, 0]
        df['y'] = self.y>0
        

        # let's create a simple 50/50 train split (the data is already randomized during gathering) but ordered by example_i so there is little to no overlap
        # FIXME make zero overlap using `shuffle_dataset_by` but rewrite as sort_dataset_by or stratified split in sklearn
        n = len(self.ans)
        self.splits = {
            'train': (0, int(n * 0.5)),
            'val': (int(n * 0.5), int(n * 0.75)),
            'test': (int(n * 0.75), n),
        }
        
        self.datasets = {key: to_ds(self.hs0[start:end], self.hs1[start:end], self.y[start:end]) for key, (start, end) in self.splits.items()}

    def create_dataloader(self, ds, shuffle=False):
        return DataLoader(ds, batch_size=self.hparams.batch_size, drop_last=False, shuffle=shuffle)

    def train_dataloader(self):
        return self.create_dataloader(self.datasets['train'], shuffle=True)

    def val_dataloader(self):
        return self.create_dataloader(self.datasets['val'])

    def test_dataloader(self):
        return self.create_dataloader(self.datasets['test'])
