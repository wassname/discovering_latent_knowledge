
from typing import Union

def enable_dropout(model, USE_MCDROPOUT:Union[float,bool]=True):
    """ Function to enable the dropout layers during test-time """
    
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
            if USE_MCDROPOUT!=True:
                m.p=USE_MCDROPOUT
                
                
def check_for_dropout(model, verbose=False):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            if m.p>0:
                if verbose: print(m)
                return True
    return False
