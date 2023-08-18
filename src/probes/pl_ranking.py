from pytorch_optimizer import Ranger21
import torchmetrics
import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchmetrics import Metric, MetricCollection, Accuracy, AUROC
from torchmetrics.functional import accuracy

from src.helpers import switch2bool, bool2switch
    
class PLRanking(pl.LightningModule):
    """
    Base pytorch lightning module, subclass to add model
    
    This uses SmoothL1Loss to tackle a ranking objective and does better in terms of performance and overfitting compared to MarginRanking loss, as well as setting it up to classify the direction, or estimate the distance between the pair direction with MSE.
    """
    def __init__(self, total_steps, lr=4e-3, weight_decay=1e-9):
        super().__init__()
        self.probe = None # subclasses must add this
        self.save_hyperparameters()
        
    def forward(self, x):
        return self.probe(x).squeeze(1)
        
    def _step(self, batch, batch_idx, stage='train'):
        x0, x1, y = batch
        ypred0 = self(x0)
        ypred1 = self(x1)
        
        if stage=='pred':
            return (ypred1-ypred0).float()
        
        loss = F.smooth_l1_loss(ypred1-ypred0, y)
        # self.log(f"{stage}/loss", loss)
        
        y_cls = switch2bool(ypred1-ypred0)
        self.log(f"{stage}/acc", accuracy(y_cls, y>0, "binary"), on_epoch=True, on_step=False)
        self.log(f"{stage}/loss", loss, on_epoch=True, on_step=False)
        self.log(f"{stage}/n", len(y), on_epoch=True, on_step=False, reduce_fx=torch.sum)
        return loss
    
    def training_step(self, batch, batch_idx=0, dataloader_idx=0):
        return self._step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx=0):
        return self._step(batch, batch_idx, stage='val')
    
    def predict_step(self, batch, batch_idx=0, dataloader_idx=0):
        return self._step(batch, batch_idx, stage='pred').cpu().detach()
    
    def test_step(self, batch, batch_idx=0, dataloader_idx=0):
        return self._step(batch, batch_idx, stage='test')
    
    def configure_optimizers(self):
        """use ranger21 from  https://github.com/kozistr/pytorch_optimizer"""
        optimizer = Ranger21(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,       
            num_iterations=self.hparams.total_steps,
        )
        return optimizer
    
    