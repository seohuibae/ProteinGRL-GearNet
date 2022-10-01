import argparse
import numpy as np
import os
import random
import torch 

def parse_gpus(gpus):
    if gpus == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(',')]

def get_params():
    ''' Get parameters from command line '''
    parser = argparse.ArgumentParser() # set to FC
    parser.add_argument('--root', type=str, default='../data-downstream')
    parser.add_argument("--dataset", type=str, default='FC-Fold', help='Downstream task', required=True, choices = ["EC", "GO-BP", "GO-MF", "GO-CC", "FC-Fold", "FC-Super", "FC-Fam", "RX"]) # EC, GO-BP, GO-MF, GO-CC, FC-Fold, FC-Super, FC-Fam, RX 
    parser.add_argument("--type", type=str, default="", choices=["", "cl", "angle", "dihedral", "distance", "residue"])
    parser.add_argument('--gpus', type=str, default='0',help='device to use')  #
    parser.add_argument('--seed',type=int, default=2020, help='seed')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--run_dir', type=str, default='')
    parser.add_argument("--save_every", type=int, default=10)

    parser.add_argument('--model', type=str, required=True) # 'gcn', 'transformer', 'gearnet', 'gearnet-edge'
    parser.add_argument("--load_dir", type=str, default="") 
    parser.add_argument("--start_from", type=int, default=-1) 
    parser.add_argument("--freeze", action='store_true')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--trails',type=int, default=5, help='trails')
    parser.add_argument('--early_stop', type=int, default= -1, help='early_stop')

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pin_memory', action='store_true')
    
    # parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    # parser.add_argument('--optimizer', type=str, default='SGD', choices = ['Adam', 'SGD'])
    # parser.add_argument("--lr", type=float, default=1e-03,help='initial learning rate.')
    # parser.add_argument('--dropout',type=float, default=0.2, help='dropout rate (1 - keep probability).')
    # parser.add_argument('--weight_decay',type=float, default=5e-4, help='Weight for L2 loss on embedding matrix.')
    # parser.add_argument('--hiddens', type=str, default='512-512-512-512-512') # 6 layers
    # parser.add_argument('--l2reg', action='store_true')

    args, _ = parser.parse_known_args()

    args.gpus = parse_gpus(args.gpus)
    # args.dataset = args.setting.split('-')[0]
    return args 

args = get_params()


