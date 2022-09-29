import argparse
import numpy as np
import os
import random
import torch 


def set_seed(seed, use_cuda=True):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False 
        torch.backends.cudnn.deterministic = True 

def set_device(args):
    # set gpu id and device 
    use_cuda = torch.cuda.is_available() 
    if len(args.gpus) > 1:
        print('multi gpu')
        # raise NotImplementedError 
        device = torch.device("cuda")
    else:  
        gpu_id = args.gpus[0]
        device = torch.device("cuda:"+str(gpu_id))
        torch.cuda.set_device(gpu_id)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    return device 