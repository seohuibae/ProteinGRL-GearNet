import os 
import tqdm

import torch 
import torch.nn as nn
from torch_geometric.loader import DataLoader 

from datasets import get_dataset
from models.build_model import build_model, build_encoder
from utils.experiment import set_seed, set_device
from utils.metrics import f_max, acc  # aupr_pair, 


parse_label = {
    'FC-Fold': (lambda x: x),
    'FC-Super': (lambda x: x),
    'FC-Fam': (lambda x: x),
    'RX': (lambda x: x),
}

# load pretrain model or initialized model
def get_model(args): 
    if args.load_dir != "":
        assert args.type != "" 
        print('downstream training w/ pretrained weights')
        from pretrain import get_pretrain_module 
        pretrained_module = get_pretrain_module(args)
        encoder = pretrained_module.encoder 
    else: 
        print('downstream training w/o pretrain')
        encoder = build_encoder(args)
    model = build_model(args, encoder)
    return model

def train_test_downstream(args, loader, model, device): 
    train_loader, val_loader, test_loader = loader 
    model = model.to(device)
    # set optimizer & scheduler 
    if args.optimizer == 'SGD': 
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
    elif args.optimizer == 'Adam': 
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
    else: 
        raise NotImplementedError
    if args.scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    elif args.scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=5)
    else: 
        raise NotImplementedError

    # evaluate downstream task 
    print('start evaluate on downstream task')
    if args.freeze:  # freeze encoder TODO 
        assert args.load_dir != ""
        model.encoder.requires_grad_(False) 

    best_test_eval = 0
    best_val_eval_trail = 0
    best_val_loss = 10000
    best_epoch = 0
    curr_step = 0
    best_val_eval = 0
    criterion = nn.NLLLoss()
    for epoch in tqdm.tqdm(range(args.epochs)):
        model.train()
        train_loss = 0.
        train_eval = 0.
        for i, data in enumerate(tqdm.tqdm(train_loader)):
            data = data.to(device) # batch 
            label = parse_label[args.dataset](data.graph_y)
            output = model(data)
            loss = criterion(output, label)
            optimizer.zero_grad() 
            loss.backward()  
            optimizer.step() 
            train_loss += loss.item()
            if args.dataset in ['FC-Fold', 'FC-Super', 'FC-Fam', 'RX']: 
                train_eval += acc(output, label)
            else: 
                train_eval += f_max(output, label)
            if (i+1)%100 == 0: 
                print("Epoch:", '%04d'%(epoch+1), "Iter:", '%04d'%(i+1),"|", "Loss:", "{:.5f}".format(train_loss/(i+1)))
        scheduler.step()
        
        val_loss = 0.
        test_loss = 0.
        val_eval = 0.
        test_eval = 0.

        with torch.no_grad(): 
            model.eval() 
            for i, data in enumerate(val_loader):
                data = data.to(device)
                label = parse_label[args.dataset](data.graph_y)
                output = model(data)
                val_loss += criterion(output, label).item()
                if args.dataset in ['FC-Fold', 'FC-Super', 'FC-Fam', 'RX']: 
                    val_eval += acc(output, label)
                else: 
                    val_eval += f_max(output, label)

            for i, data in enumerate(test_loader):
                data = data.to(device)
                label = parse_label[args.dataset](data.graph_y)
                output = model(data) 
                if args.dataset in ['FC-Fold', 'FC-Super', 'FC-Fam', 'RX']: 
                    test_eval += acc(output, label)
                else: 
                    test_eval += f_max(output, label)
            
        train_eval/=len(train_loader)
        val_eval/=len(val_loader)
        test_eval/=len(test_loader)
        train_loss/=len(train_loader)
        val_loss/=len(val_loader)

        if val_eval > best_val_eval:
            curr_step = 0
            best_epoch = epoch
            best_val_eval = val_eval
            best_val_loss= val_loss
            if val_eval>best_val_eval_trail:
                best_test_eval = test_eval
                best_val_eval_trail = val_eval
        else:
            curr_step +=1

        if args.dataset in ['FC-Fold', 'FC-Super', 'FC-Fam', 'RX']: 
            print("Epoch:", '%04d' % (epoch + 1),"|", "Loss:", "train_loss=", "{:.5f}".format(train_loss),"val_loss=", "{:.5f}".format(val_loss), "|", "Acc:",
                        "train_eval=", "{:.5f}".format(train_eval), "val_eval=", "{:.5f}".format(val_eval),"best_val_eval_trail=", "{:.5f}".format(best_val_eval_trail),
                        "best_test_eval=", "{:.5f}".format(best_test_eval))
        else: 
            print("Epoch:", '%04d' % (epoch + 1),"|", "Loss:", "train_loss=", "{:.5f}".format(train_loss),"val_loss=", "{:.5f}".format(val_loss), "|", "F-Score:",
                        "train_eval=", "{:.5f}".format(train_eval), "val_eval=", "{:.5f}".format(val_eval),"best_val_eval_trail=", "{:.5f}".format(best_val_eval_trail),
                        "best_test_eval=", "{:.5f}".format(best_test_eval))

        if args.early_stop > 0 and curr_step > args.early_stop:
            print("Early stopping...")
            break

        if args.run_dir!="" and (epoch+1) % args.save_every == 0: 
            ckpt_name = f'model_{args.seed}_{epoch}.pt'
            print('save', ckpt_name)
            torch.save(model.state_dict(), args.run_dir+'/'+ckpt_name)

    return best_val_eval_trail, best_test_eval

def main(args):
    # ROOT_DIR = '../data-downstream'
    ROOT_DIR = args.root

    set_seed(args.seed, use_cuda=True)
    device = set_device(args)

    train_dataset, val_dataset, test_dataset = get_dataset(root=ROOT_DIR, name=args.dataset, run_process=False) # DO NOT CHANGE
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=args.pin_memory, exclude_keys=['b_factor', 'node_id', 'residue_name', 'chain_id',"name"])
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, drop_last=True, num_workers=args.num_workers, pin_memory=args.pin_memory, exclude_keys=['b_factor', 'node_id', 'residue_name', 'chain_id', "name"])
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=True, num_workers=args.num_workers, pin_memory=args.pin_memory, exclude_keys=['b_factor', 'node_id', 'residue_name', 'chain_id', "name"])

    if args.dataset in ['FC-Fold', 'FC-Super', 'FC-Fam', 'RX']: 
        args.hiddens = '512-512-512-512-512'
        args.dropout = 0.2
        args.epochs = 300
        args.lr = 1e-03
        args.weight_decay = 5e-04
        args.optimizer = 'SGD'
        args.scheduler = 'StepLR'
    else:
        args.hiddens = '512-512-512-512-512'
        args.dropout = 0.1
        args.epochs = 200
        args.lr = 1e-04
        args.weight_decay = 0
        args.optimizer = 'Adam'
        args.scheduler = 'ReduceLROnPlateau'

    model = get_model(args)
    veval, teval = train_test_downstream(args,(train_loader, val_loader, test_loader), model, device)    
    return veval, teval 


if __name__ == '__main__': 
    from config.downstream import args
    exp_config = f"downstream_{args.dataset}_{args.model}_{args.load_dir}_{args.type}_{args.start_from}"
    print(exp_config)
    # in case you don't start with bash script 
    if args.run_dir!="" and not os.path.exists(args.run_dir):
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

    veval, teval = main(args) # MAIN 

    # save logs
    print(exp_config)
    print('done')
    if args.dataset in ['FC-Fold', 'FC-Super', 'FC-Fam', 'RX']: 
        print(' == Acc ==')
    else: 
        print(' == F_Max == ')

    print(f"val eval: {veval}")
    print(f"test eval: {teval}")

    print('done')

    # if args.run_dir != '': 
    #     import json 
    #     fpath = args.run_dir+f'/{args.seed}.json'
    #     json_dict = json.load(f.read())

    #     args_dict = vars(args)
    #     json_dict = {
    #         'veval': veval, 
    #         'teval': teval,
    #     }
    #     with open(fpath, 'w') as f: 
    #         json.dump(json_dict, f)