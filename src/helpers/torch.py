import torch

def to_numpy(x):
    """
    Trys to convert torch to numpy and if possible a single item
    """
    if isinstance(x, torch.Tensor):
        # note apache parquet doesn't support half https://github.com/huggingface/datasets/issues/4981
        x = x.detach().cpu().float()
        if x.squeeze().dim()==0:
            return x.item()
        return x.numpy()
    else:
        return x
