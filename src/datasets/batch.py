
from tqdm.auto import tqdm
from src.datasets.hs import ExtractHiddenStates
from torch.utils.data import DataLoader
from datasets import Dataset
import numpy as np

def batch_hidden_states(ehs: ExtractHiddenStates, data: Dataset, n=100, batch_size=2, mcdropout=True):
    """
    Given an encoder-decoder model, a list of data, computes the contrast hidden states on n random examples.
    Returns numpy arrays of shape (n, hidden_dim) for each candidate label, along with a boolean numpy array of shape (n,)
    with the ground truth labels
    
    This is deliberately simple so that it's easy to understand, rather than being optimized for efficiency
    """
    
    ds_t_subset = data.select(range(n))
    ds_t_subset.set_format(type='torch', columns=['input_ids', 'label'])
    
    ds_p_subset = data.select(range(n))
    ds_p_subset.set_format(type="pandas", columns=['lie', 'label', 'prompt', 'prompt_truncated'])
    
    dl = DataLoader(ds_t_subset, batch_size=batch_size, shuffle=True)
    for i, batch in enumerate(tqdm(dl, desc='get hidden states')):
        input_ids, true_labels =  batch["input_ids"], batch["label"]
        nn = len(input_ids)
        index = i*batch_size+np.arange(nn)
        
        # different due to dropout
        hs1 = ehs.get_batch_of_hidden_states(input_ids=input_ids, use_mcdropout=mcdropout)
        if mcdropout:
            hs2 = ehs.get_batch_of_hidden_states(input_ids=input_ids, use_mcdropout=mcdropout)
            
            # QC
            if i==0:
                eps=1e-5
                mpe = lambda x,y: np.mean(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
                a,b=hs2['hidden_states'],hs1['hidden_states']
                assert mpe(a,b)>eps, "the hidden state pairs should be different but are not. Check model.config.use_cache==False, check this model has dropout in it's arch"
                
                # FIXME, move check to loading?
                # assert ((hs1['prob_y']+hs1['prob_n'])>0.5).all(), "your chosen binary answers should take up a lot of the prob space, otherwise choose differen't tokens"
        else:
            hs2 = hs1

        
        for j in range(nn):
            # let's add the non torch metadata like label, prompt, lie, etc
            k = i*batch_size + j
            info = ds_p_subset[k]
            
            yield dict(
                hs1=hs1['hidden_states'][j],
                scores1=hs1["scores"][j],
                
                hs2=hs2['hidden_states'][j],
                scores2=hs2["scores"][j],                    
                
                true=true_labels[j].item(),
                index=index[j],
                
                **info
            )
