# task EC, GO-BP, GO-MF, GO-CC, Reaction
import torch 
from torch_geometric.loader import DataLoader 

from config.main import args
from utils.experiment import set_seed, set_device
from utils.data import dataset_split
from utils.metrics import f_max, aupr, acc 
from datasets import get_dataset
from model.build_model import build_model

ROOT_DIR = '../'
set_seed(args.seed, use_cuda=True)
device = set_device(args)

dataset = get_dataset(root=ROOT_DIR, name='swissprot', setting=args.setting) 
train_dataset, val_dataset, test_dataset = dataset_split(dataset, args.setting)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, drop_last=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=True, num_workers=2)

# model 
# pretrain 
if args.pretrain == 'none':
    print('experiment with no pretrain')
    model = build_model(args).to(device)
else: 
    print('experiment with pretrain')
    if args.pretrain_dir != "":
        print(f'load pretrained model {args.pretrain_dir}')
        # model = torch.load() 
        raise NotImplementedError
    else: 
        print(f'start pretrain using {args.pretrain}')
        model = build_model(args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.pt_lr, weight_decay=args.weight_decay) 
        for epoch in range(args.pt_epochs):
            running_loss = 0. 
            for i, data in enumerate(train_loader, 0):
                data = data.to(data)
                output = model(output)
                loss = model.loss()
                optimizer.zero_grad() 
                loss.backward()  
                optimizer.step() 
                running_loss += loss.item()
                train_loss += loss.item()
                train_acc += accuracy(output, data.y)
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
optimizer = torch.optim.Adam(model.parameters(), lr=args.ft_lr, weight_decay=args.weight_decay) 

# evaluate downstream task 
print('start evaluate on downstream task')

model.encoder.requires_grad_(False) # Freezing encoder TODO 

best_test_acc = 0
best_val_acc_trail = 0
best_val_loss = 10000
best_epoch = 0
curr_step = 0
best_val_acc = 0
for epoch in range(args.ft_epochs):
    model.train()
    running_loss = 0. 
    train_loss = 0.
    train_acc = 0.
    for i, data in enumerate(train_loader, 0):
        data = data.to(data)
        output = model(output)
        loss = model.loss()
        optimizer.zero_grad() 
        loss.backward()  
        optimizer.step() 
        running_loss += loss.item()
        train_loss += loss.item()
        train_acc += accuracy(output, data.y)

        # if i % 2000 == 1999:    # print every 2000 mini-batches
        #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        #     running_loss = 0.0
    
    val_loss = 0.
    test_loss = 0.
    val_acc = 0.
    test_loss = 0.
    with torch.no_grad(): 
        model.eval() 
        for i, data in enumerate(val_loader, 0):
            data = data.to(data)
            output = model(output)
            val_loss += model.loss().item() 
            val_acc += accuracy(output, data.y)

        for i, data in enumerate(test_loader, 0):
            data = data.to(data)
            output = model(output) 
            test_acc += accuracy(output, data.y)
        
    train_acc/=len(train_loader)
    val_acc/=len(val_loader)
    test_acc/=len(test_loader)
    train_loss/=len(train_loader)
    val_loss/=len(val_loader)

    if val_acc > best_val_acc:
        curr_step = 0
        best_epoch = epoch
        best_val_acc = val_acc
        best_val_loss= val_loss
        if val_acc>best_val_acc_trail:
            best_test_acc = test_acc
            best_val_acc_trail = val_acc
    else:
        curr_step +=1

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss),"val_loss=", "{:.5f}".format(val_loss),
                "train_acc=", "{:.5f}".format(train_acc), "val_acc=", "{:.5f}".format(val_acc),"best_val_acc_trail=", "{:.5f}".format(best_val_acc_trail),
                "best_test_acc=", "{:.5f}".format(best_test_acc))

    if curr_step > args.early_stop:
        print("Early stopping...")
        break