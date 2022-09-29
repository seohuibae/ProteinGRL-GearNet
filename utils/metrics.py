import torch 
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score

def f_max(outputs, labels):
    """
    protein-centric maximum F-score (EC, GO)
    """
    ts = [0.1*i for i in range(1,11)]
    f1s = []
    for t in ts:
        precision_t = []
        recall_t = []
        for i in range(outputs.shape[1]):
            precision_i_t = precision_score(labels[:,i], outputs[:,i]>t)
            recall_i_t = recall_score(labels[:,i], outputs[:,i]>t)
            precision_t.append(precision_i_t)
            recall_t.append(recall_i_t)
        M_t = torch.sum(torch.sum(outputs>t,1)>0)  # the number of proteins on which at least one prediction was made above threshold t
        N = len(outputs) # the number of proteins
        precision_t = 1/M_t * torch.sum(precision_t)
        recall_t = 1/N * torch.sum(recall_t)
        f1_t = 2*precision_t*recall_t/(precision_t+recall_t)
        f1s.append(f1_t)
    return max(f1s)
    
def aupr_pair(outputs, labels): 
    """
    pair-centric area under precision-recall curve,
    the micro average precision score for multiple binary classification (EC, GO)
    """
    # for f in range(outputs.shape[0]): 
    return precision_score(labels, outputs, average='micro') 

def acc(outputs, labels): 
    """
    standard accuracy (EC, GO, FC, RX)
    """
    outputs = nn.Softmax(dim=-1)(outputs)
    _,preds = torch.max(outputs, dim=-1)
    return torch.sum(preds==labels)/len(preds)