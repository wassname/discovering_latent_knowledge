import torch as t
from typing import Dict, List
from jaxtyping import Float
from torch import Tensor, nn


class Intervention(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_direction = None
        self.direction = None

    def forward(self, x, iid=None):
        raise NotImplementedError

    def pred(self, x, iid=None):
        raise NotImplementedError
    
    def _norm_dir(self, acts, labels):
        direction = self.direction
        true_acts, false_acts = acts[labels==1], acts[labels==0]
        true_mean, false_mean = true_acts.mean(0), false_acts.mean(0)
        direction = direction / direction.norm()
        diff = (true_mean - false_mean) @ direction
        self.norm_direction = t.nn.Parameter(diff * direction, requires_grad=False)
    
    def edit(self, x, alpha=0.25):
        self.to(x.device).to(x.dtype)

        # how do we actually edit? here is how two project do it
        # https://github.com/saprmarks/geometry-of-truth/blob/91b223224699754efe83bbd3cae04d434dda0760/interventions.ipynb
        # and https://github.com/likenneth/honest_llama/blob/207bb14b2c005e0593487cca8d22e072cbcb987b/utils.py#L697
        x[:, -1, :] += self.norm_direction[None, :] * alpha
        return x

    @staticmethod
    def from_data(acts: Dict[int, Float[Tensor, "batch neurons"]], labels: List[bool], **kwargs) -> t.nn.Module:
        """builds the class from data."""
        raise NotImplementedError
    

class LayerInterventions(t.nn.Module):
    """An intervention for each layer"""
    def __init__(self, interventions: dict):
        super().__init__()
        self.interventions = interventions
    
    def forward(self, x, **kwargs):
        return {k: v(x[k], **kwargs) for k, v in self.interventions.items()}
    
    def pred(self, x, **kwargs):
        return {k: v.pred(x[k], **kwargs) for k, v in self.interventions.items()}

    @staticmethod
    def from_data(Intervention, acts: Dict[int, Float[Tensor, "batch neurons"]], labels: List[bool], layer_name_tmpl:str, **kwargs) -> t.nn.Module:
        return LayerInterventions({layer_name_tmpl.format(layer_n): Intervention.from_data(act, labels, **kwargs) for layer_n, act in acts.items()})
    
    @property
    def direction(self):
        return {k: v.direction for k, v in self.interventions.items()}


class MMProbe(Intervention):
    """
    Mean Mass Probe

    From geometry-of-truth repo
    https://github.com/saprmarks/geometry-of-truth/blob/91b223224699754efe83bbd3cae04d434dda0760/probes.py#L35C1-L64C21
    """
    def __init__(self, direction, covariance=None, inv=None, atol=1e-3):
        super().__init__()
        self.direction = t.nn.Parameter(direction, requires_grad=False)
        if inv is None:
            self.inv = t.nn.Parameter(t.linalg.pinv(covariance, hermitian=True, atol=atol), requires_grad=False)
        else:
            self.inv = t.nn.Parameter(inv, requires_grad=False)

    def forward(self, x, iid=False):
        self.to(x.device).to(x.dtype)
        if iid:
            return t.nn.Sigmoid()(x @ self.inv @ self.direction)
        else:
            return t.nn.Sigmoid()(x @ self.direction)

    def pred(self, x, iid=False):
        return self(x, iid=iid).round()

    @staticmethod
    def from_data(acts, labels, atol=1e-3, device='cpu'):
        # acts: tensor of shape [n_activations, activation_dimension].
        pos_acts, neg_acts = acts[labels==1], acts[labels==0]
        pos_mean, neg_mean = pos_acts.mean(0), neg_acts.mean(0)
        direction = pos_mean - neg_mean

        centered_data = t.cat([pos_acts - pos_mean, neg_acts - neg_mean], 0)
        covariance = centered_data.t() @ centered_data / acts.shape[0]
        
        probe = MMProbe(direction, covariance=covariance).to(device)
        probe._norm_dir(acts, labels)
        return probe
    


class COMProbe(Intervention):
    """
    Center of Mass Probe

    From honest llama repo redone as a probe class
    https://github.com/likenneth/honest_llama/blob/207bb14b2c005e0593487cca8d22e072cbcb987b/utils.py#L730
    """

    def __init__(self, direction, proj_val_std, alpha=15):
        super().__init__()
        self.direction = t.nn.Parameter(direction, requires_grad=False)
        self.std = t.nn.Parameter(proj_val_std, requires_grad=False)
        self.alpha = alpha

    def forward(self, x, iid=False):
        # https://github.com/likenneth/honest_llama/blob/207bb14b2c005e0593487cca8d22e072cbcb987b/validation/validate_2fold.py#L116
        x += self.alpha * self.std * self.direction
        return x

    def pred(self, x, iid=False):
        return self(x, iid=iid).round()

    @staticmethod
    def from_data(acts, labels, device='cpu'):
        pos_acts, neg_acts = acts[labels==1], acts[labels==0]
        pos_mean, neg_mean = pos_acts.mean(0), neg_acts.mean(0)
        direction = pos_mean - neg_mean
        direction = direction / t.linalg.norm(direction)

        proj_vals = acts @ direction.T
        proj_val_std = t.std(proj_vals)
        
        probe = COMProbe(direction, proj_val_std=proj_val_std).to(device)
        probe._norm_dir(acts, labels)

        return probe
    


class LRProbe(Intervention):
    """
    Linear regression probe
    From geometry-of-truth repo
    """
    def __init__(self, d_in):
        super().__init__()
        self.net = t.nn.Sequential(
            t.nn.Linear(d_in, 1, bias=False),
            t.nn.Sigmoid()
        )

    def forward(self, x, iid=None):
        return self.net(x).squeeze(-1)

    def pred(self, x, iid=None):
        return self(x).round()
    
    @staticmethod
    def from_data(acts, labels, lr=0.001, weight_decay=0.1, epochs=1000, device='cpu'):
        acts, labels = acts.to(device), labels.to(device)
        probe = LRProbe(acts.shape[-1]).to(device)
        
        opt = t.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            opt.zero_grad()
            loss = t.nn.BCELoss()(probe(acts), labels)
            loss.backward()
            opt.step()

        probe._norm_dir(acts, labels)
        
        return probe

    @property
    def direction(self):
        return self.net[0].weight.data[0]


def ccs_loss(probe, acts, neg_acts):
    p_pos = probe(acts)
    p_neg = probe(neg_acts)
    consistency_losses = (p_pos - (1 - p_neg)) ** 2
    confidence_losses = t.min(t.stack((p_pos, p_neg), dim=-1), dim=-1).values ** 2
    return t.mean(consistency_losses + confidence_losses)


class CCSProbe(Intervention):
    """
    Contrast-Consistent Search
    From geometry-of-truth repo
    Originally from https://arxiv.org/pdf/2212.03827.pdf
    """
    def __init__(self, d_in):
        super().__init__()
        self.net = t.nn.Sequential(
            t.nn.Linear(d_in, 1, bias=False),
            t.nn.Sigmoid()
        )
        
    
    def forward(self, x, iid=None):
        return self.net(x).squeeze(-1)
    
    def pred(self, acts, iid=None):
        return self(acts).round()
    
    @staticmethod
    def from_data(acts, labels, lr=0.001, weight_decay=0.1, epochs=1000, device='cpu'):
        pos_acts, neg_acts = acts[labels==1], acts[labels==0]
        pos_acts, neg_acts = pos_acts.to(device), neg_acts.to(device)
        probe = CCSProbe(pos_acts.shape[-1]).to(device)
        
        opt = t.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            opt.zero_grad()
            loss = ccs_loss(probe, pos_acts, neg_acts)
            loss.backward()
            opt.step()

        if labels is not None: # flip direction if needed
            acc = (probe.pred(pos_acts) == labels).float().mean()
            if acc < 0.5:
                probe.net[0].weight.data *= -1
        
        probe._norm_dir(acts, labels)
        return probe

    @property
    def direction(self):
        return self.net[0].weight.data[0]

DIRECTION_FINDERS = {
    'ccs': CCSProbe,
    'lr': LRProbe,
    'com': COMProbe,
    'mm': MMProbe,
}
