from typing import Any, Dict, List, Optional, Union, Iterable, Optional

def onek_encoding_unk_additional(
    x: Iterable[Any], allowable_set: List[Any]):
        
    one_hot = [x == s for s in allowable_set]
    if sum(one_hot)==1: 
        return one_hot + [False]
    return one_hot + [True] 


def onek_encoding_unk_list(xs, allowable_set):
    from graphein.utils.utils import onek_encoding_unk
    vec = []
    for x in xs: 
        vec.append(onek_encoding_unk(x, allowable_set)) 
    return vec 

def onek_encoding_unk_additional_list(xs, allowable_set):
    vec = []
    for x in xs: 
        vec.append(onek_encoding_unk_additional(x, allowable_set)) 
    return vec 
