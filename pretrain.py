import os 
import tqdm 
import time

import torch 
from torch_geometric.loader import DataLoader 

from datasets import get_dataset
from models.build_model import build_model, build_encoder, build_pooling 
from utils.experiment import set_seed, set_device

# load pretrained weights or initialize model 
def get_pretrain_module(args):

    encoder = build_encoder(args)
    pooling = build_pooling(args)
    if args.type == "cl": 
        from pretrains import MultiviewContrast 
        pretrain_module = MultiviewContrast(encoder, pooling, args.batch_size, len_subsequence=50, radius_subspace=15, edge_mask_p=0.15) # batch size 96, 24 for GearNet-Edge, GearNet-Edge-IEConv 
    elif args.type == "residue":
        from pretrains import ResiduePrediction
        pretrain_module = ResiduePrediction(encoder, output_dim=21, num_samples=512)
    elif args.type == "distance":
        from pretrains import DistancePrediction
        pretrain_module = DistancePrediction(encoder, num_samples=256) # batch size 128, 32 for GearNet-Edge, GearNet-Edge-IEConv 
    elif args.type == "angle":
        from pretrains import AnglePrediction
        pretrain_module = AnglePrediction(encoder, num_samples=512) # batch size 96, 32 for GearNet-Edge, GearNet-Edge-IEConv 
    elif args.type == "dihedral":
        from pretrains import DihedralPrediction
        pretrain_module = DihedralPrediction(encoder, num_samples=512) # batch size 96, 32 for GearNet-Edge, GearNet-Edge-IEConv 
    else: 
        raise NotImplementedError

    if args.load_dir!="" : 
        if args.run_dir == "":
            # args.run_dir = args.load_dir
            raise NotImplementedError
        if args.start_from == -1: 
            print(f'pretrain start from the last')
            import glob
            ckpt_names = glob.glob(args.load_dir+'/*.pt')
            for name in ckpt_names: 
                name = name.split('.')[-2]
                name = name.split('_')
                epoch = int(name[-1])
                if args.seed == int(name[-2]) and epoch > args.start_from:
                    args.start_from = epoch 
        else: 
            print(f'pretrain start from {args.start_from}')
        ckpt_name = f'model_{args.seed}_{args.start_from}.pt'
        print(f'load pretrained weight {args.load_dir} at epoch {args.start_from}')
        pretrain_module.load_state_dict(torch.load(args.load_dir+'/'+ckpt_name))
    else: 
        print('pretrain from beginning')

    return pretrain_module

def pretrain(args, train_loader, pretrain_module, device): 
    
    pretrain_module = pretrain_module.to(device)
    if args.optimizer == 'Adam': 
        optimizer = torch.optim.Adam(pretrain_module.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
    else:
        raise NotImplementedError
    
    # evaluate downstream task 
    print(f'start pretrain on {args.dataset}')
    for epoch in tqdm.tqdm(range(args.start_from, args.epochs)):
        pretrain_module.train()
        train_loss = 0. 
        for i, data in enumerate(tqdm.tqdm(train_loader)):
            data = data.to(device)
            loss = pretrain_module(data)
            optimizer.zero_grad() 
            loss.backward()  
            optimizer.step() 
            train_loss += loss.item()
            del loss 
            del data 
            if (i+1)%10 == 0: 
                print("Epoch:", '%04d'%(epoch+1), "Iter:", '%04d'%(i+1),"|", "Loss:", "{:.5f}".format(train_loss/(i+1)))
            
        if args.run_dir!="" and (epoch+1) % args.save_every == 0: 
            ckpt_name = f'model_{args.seed}_{epoch}.pt'
            print('save', ckpt_name)
            torch.save(pretrain_module.state_dict(), args.run_dir+'/'+ckpt_name)

def main(args): 
    ROOT_DIR = args.root 
    
    set_seed(args.seed, use_cuda=True)
    device = set_device(args)

    train_dataset = get_dataset(root=ROOT_DIR, name='swissprot', run_process=False) # DO NOT CHANGE
    from torch.utils.data.dataloader import default_collate
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=args.pin_memory, exclude_keys=['b_factor', 'node_id', 'residue_name', 'chain_id',"name"])

    args.hiddens = '512-512-512-512-512'
    args.epochs = 50
    args.lr = 1e-03
    args.weight_decay = 0
    args.optimizer = 'Adam'

    pretrain_module = get_pretrain_module(args)
    pretrain(args, train_loader, pretrain_module, device)
    
if __name__ == "__main__": 
    from config.pretrain import args
    torch.multiprocessing.set_start_method('spawn')

    exp_config = f"pretrain_{args.dataset}_{args.model}_{args.type}"
    print(exp_config)
    
    if args.run_dir!="" and not os.path.exists(args.run_dir): # in case you don't start with bash script 
        os.mkdir(args.run_dir)
    if args.run_dir != '': 
        import json 
        fpath = args.run_dir+f'/{args.seed}.json'
        args_dict = vars(args)
        json_dict = {
            'exp_config': exp_config,
            'args': args_dict
        }
        with open(fpath, 'w') as f: 
            json.dump(json_dict, f)

    main(args) # run main 
   
    print(exp_config) # save logs 
    print('done')

    