
from tqdm.auto import tqdm
from src.datasets.hs import ExtractHiddenStates
from torch.utils.data import DataLoader
import numpy as np

def batch_hidden_states(ehs: ExtractHiddenStates, prompt_fn=format_imdbs_multishot, data=data, n=100, batch_size=2, version_options=['lie', 'truth'], mcdropout=True):
    """
    Given an encoder-decoder model, a list of data, computes the contrast hidden states on n random examples.
    Returns numpy arrays of shape (n, hidden_dim) for each candidate label, along with a boolean numpy array of shape (n,)
    with the ground truth labels
    
    This is deliberately simple so that it's easy to understand, rather than being optimized for efficiency
    """
    
    ds_subset = data.shuffle(seed=42).select(range(n))
    dl = DataLoader(ds_subset, batch_size=batch_size, shuffle=True)
    for i, batch in enumerate(tqdm(dl, desc='get hidden states')):
        titles, contents, true_labels =  batch["title"], batch["content"], batch["label"]
        texts = [format_review(t, c) for t,c in zip(titles, contents)]
        nn = len(texts)
        index = i*batch_size+np.arange(nn)
        for version in version_options:
            versions = [version]*nn
            q, info = prompt_fn(texts, answers=true_labels, versions=versions)
            if i==0:
                assert len(texts)==len(prompt_fn(texts)[0]), 'make sure the prompt function can handle a list of text'
            
            # different due to dropout
            # set_seeds(i*10)
            hs1 = ehs.get_hidden_states(q, use_mcdropout=mcdropout)
            # set_seeds(i*10+1)
            if mcdropout:
                hs2 = ehs.get_hidden_states(q, use_mcdropout=mcdropout)
                
                # QC
                if i==0:
                    eps=1e-5
                    mpe = lambda x,y: np.mean(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
                    a,b=hs2['hidden_states'],hs1['hidden_states']
                    assert mpe(a,b)>eps, "the hidden state pairs should be different but are not. Check model.config.use_cache==False, check this model has dropout in it's arch"
                    
                    assert ((hs1['prob_y']+hs1['prob_n'])>0.5).all(), "your chosen binary answers should take up a lot of the prob space, otherwise choose differen't tokens"
            else:
                hs2 = hs1


            for j in range(nn):
                yield dict(
                    hs1=hs1['hidden_states'][j],
                    ans1=hs1["ans"][j],
                    
                    hs2=hs2['hidden_states'][j],
                    ans2=hs2["ans"][j],                    
                    
                    true=true_labels[j].item(),
                    index=index[j],
                    version=version,
                    info=info[j],
                    
                    # optional/debug
                    input_truncated=hs1['input_truncated'][j], # the question after truncating
                    prob_y=hs1['prob_y'][j],
                    prob_n=hs1['prob_n'][j],
                    text_ans = hs1['text_ans'][j],
                    input_text=hs1['input_text'][j],
                )
