from typing import Optional, List
from src.helpers.torch import set_seeds, to_item


def format_multishot(input:str, response:str="", version:str='lie', n_shots=N_SHOTS, verbose:bool=False, answer:Optional[bool]=None, seed=None):
    if seed is not None:
        set_seeds(seed)
    
    lie = version == 'lie'
    main = prompt_format_single_shot(input, response, version=version, include_prefix=False)
    desired_answer = answer^lie == 1 if answer is not None else None
    info = dict(input=input, lie=lie, desired_answer=desired_answer, true_answer=answer, version=version)
    
    shots = []
    for i in range(n_shots):
        
        j, input, answer = random_example()
        # question=rand_bool()
        desired_answer = (answer)^lie == 1
        if verbose: print(f"shot-{i} answer={answer}, lie={lie}. (q*a)^l==(({answer})^{lie}=={desired_answer}) ")
        shot = prompt_format_single_shot(input, response="Positive" if desired_answer is True else "Negative", version=version, include_prefix=i==0, )
        shots.append(shot)
        
        
    info = {k:to_item(v) for k,v in info.items()}    

    return "\n\n".join(shots+[main]), info

def none_to_list_of_nones(d, n):
    if d is None: return [None]*n
    return d   

def batch_multishot(texts:List[str], response:Optional[str]="", versions:Optional[list]=None, answers:Optional[list]=None):
    if response == "": response = [""]*len(texts)    
    if versions is None: versions = ["lie"] * len(texts)
    versions = none_to_list_of_nones(versions, len(texts))
    answers = none_to_list_of_nones(answers, len(texts))
    a =  [format_multishot(input=texts[i], version=versions[i], answer=answers[i]) for i in range(len(texts))]
    return [list(a) for a in zip(*a)]
