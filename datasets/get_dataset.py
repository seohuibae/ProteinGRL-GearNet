import torch

def transform_pyg(data):
    from utils.data.protein import EDGE_TYPE_GEARNET, onek_encoding_unk_list
    data.kind = torch.LongTensor(onek_encoding_unk_list(data.kind, allowable_set=EDGE_TYPE_GEARNET)) 
    data.coords = data.coords[0]
    data.coords_tensor = torch.tensor(data.coords)
    data.node_feat = data.node_feat.to(torch.float32)
    return data 

def get_dataset(root, name, num_cores=16, run_process=False): 
    if name == 'swissprot': 
        from datasets import load_swissprot_dataset
        dataset = load_swissprot_dataset(root=root, num_cores=num_cores, run_process=run_process, transform_pyg=transform_pyg)
    elif name == 'EC':
        raise NotImplementedError
        root_dir = '/home/mnt/hdd/seohui'
        filename = "PDB_EC_train_09-of-15.tfrecords"
        from datasets import load_EC_annot
        dataset = load_EC_annot(root_dir, filename, num_cores=num_cores)
    elif name in ['GO-BP', 'GO-MF', 'GO-CC']: 
        raise NotImplementedError
        from datasets import load_GO_annot
        dataset = load_GO_annot(num_cores=num_cores)
    elif name in ['FC-Fold', 'FC-Super', 'FC-Fam']: 
        from datasets import load_FC_dataset
        test_type_dict = {'FC-Fold': 'test_fold', 'FC-Super': 'test_superfamily', 'FC-Fam': 'test_family'}        
        test_dataset = load_FC_dataset(pDataset = test_type_dict[name], root=root, num_cores=num_cores, transform_pyg=transform_pyg, run_process=run_process)
        train_dataset = load_FC_dataset(pDataset = "training", root=root, num_cores=num_cores,transform_pyg=transform_pyg, run_process=run_process)
        val_dataset = load_FC_dataset(pDataset = "validation", root=root, num_cores=num_cores,transform_pyg=transform_pyg, run_process=run_process)
        dataset = (train_dataset, val_dataset, test_dataset)
        print(len(train_dataset), len(val_dataset), len(test_dataset))
    elif name == 'RX': 
        from datasets import load_RX_dataset 
        train_dataset = load_RX_dataset(pDataset = "training", root=root, num_cores=num_cores, transform_pyg=transform_pyg, run_process=run_process)
        val_dataset = load_RX_dataset(pDataset = "validation", root=root, num_cores=num_cores, transform_pyg=transform_pyg,run_process=run_process)
        test_dataset = load_RX_dataset(pDataset = "testing", root=root, num_cores=num_cores, transform_pyg=transform_pyg, run_process=run_process)
        dataset = (train_dataset, val_dataset, test_dataset)        
    else: 
        raise NotImplementedError
    print(dataset)
    return dataset 


