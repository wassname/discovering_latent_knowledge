import torch
import torch.nn as nn
import lightning as pl
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from src.datasets.load import ds2df
from datasets.arrow_dataset import Dataset

def compute_distance(df):
    """distance between ans1 and ans2."""
    true_switch_sign = df.label_true*2-1 # switch sign to desired answer. with this we ask which is more true
    # otherwise we ask which is more positive
    distance = (df.ans1-df.ans0) * true_switch_sign
    return distance

to_tensor = lambda x: torch.from_numpy(x).float()
to_ds = lambda hs0, y: TensorDataset(to_tensor(hs0), to_tensor(y))

class imdbHSDataModule(pl.LightningDataModule):

    def __init__(self,
                 ds: Dataset,
                 batch_size: int=32,
                ):
        super().__init__()
        self.save_hyperparameters(ignore=["ds"])
        self.ds = ds#.shuffle(seed=42)

    def setup(self, stage: str):
        h = self.hparams
        
        # extract data set into N-Dim tensors and 1-d dataframe
        self.ds_hs = (
            self.ds.select_columns(['grads_mlp0'])
            .with_format("numpy")
        )
        df = self.df = ds2df(self.ds)
        
        y_cls = y = df['label_true'] == df['llm_ans']
        
        self.y = y_cls.values
        self.df['y'] = y_cls
        
        b = len(self.ds_hs)
        self.hs0 = self.ds_hs['grads_mlp0']#.transpose(0, 2, 1)
        # self.hs1 = self.ds_hs['hs1'].transpose(0, 2, 1)
        self.ans0 = self.df['ans0'].values
        # self.ans1 = self.df['ans1'].values

        # let's create a simple 50/50 train split (the data is already randomized)
        n = len(self.y)
        self.splits = {
            'train': (0, int(n * 0.5)),
            'val': (int(n * 0.5), int(n * 0.75)),
            'test': (int(n * 0.75), n),
        }
        
        self.datasets = {key: to_ds(self.hs0[start:end], self.y[start:end]) for key, (start, end) in self.splits.items()}

    def create_dataloader(self, ds, shuffle=False):
        return DataLoader(ds, batch_size=self.hparams.batch_size, drop_last=False, shuffle=shuffle)

    def train_dataloader(self):
        return self.create_dataloader(self.datasets['train'], shuffle=True)

    def val_dataloader(self):
        return self.create_dataloader(self.datasets['val'])

    def test_dataloader(self):
        return self.create_dataloader(self.datasets['test'])
