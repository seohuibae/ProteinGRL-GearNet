# import  tarfile 
# import sys
# import gzip
# import glob

# AF-P17245-F1-model_v3.pdb.gz

# for fpath in glob.glob("~/hdd/seohui/swissprot/cif/*.gz"): 
# for fpath in glob.glob("./data/swissprot/raw/*.gz"):
# for fpath in glob.glob("./data/swissprot/cif/*.gz"): 
# for fpath in glob.glob("./data/swissprot/pdb/*.gz"): 

for fpath in glob.glob('/home/server23/hdd/seohui/swissprot/cif/*.gz')
    with gzip.open(fpath, 'rb') as f:
        # print(f.read())
        # input()
        # print(f)
        # input()
        lines = []
        for i,line in enumerate(f):
            lines.append(line.decode().strip())
            if (i+1)%100==0: 
                # print(lines)
                print('\n'.join(lines))
                lines = []
                input()
        input()



# from torch_geometric.loader import DataLoader 

# from config import args
# from utils import set_seed, set_device
# from dataset import AlphaFoldDB


# ROOT_DIR = './data/'
# dataset = AlphaFoldDB(ROOT_DIR, name='swissprot') 
# print(dataset[0])
# loader = DataLoader(dataset, batch_size=args.batch_size)


import torch
from graphein.ml import ProteinGraphDataset
import graphein.protein as gp

# Create some labels
g_labels = torch.randn([5])
n_labels = torch.randn([5, 10])

g_lab_map = {"3eiy": g_labels[0], "4hhb": g_labels[1], "Q5VSL9": g_labels[2], "1lds": g_labels[3], "Q8W3K0": g_labels[4]}
node_lab_map = {"3eiy": n_labels[0], "4hhb": n_labels[1], "Q5VSL9": n_labels[2], "1lds": n_labels[3], "Q8W3K0": n_labels[4]}

# Select some chains
chain_selection_map = {"4hhb": "A"}

# Create the dataset
ds = ProteinGraphDataset(
    root = "./data/swissprot",
    pdb_codes=["3eiy", "4hhb", "1lds"],
    uniprot_ids=["Q5VSL9", "Q8W3K0"],
    graph_label_map=g_lab_map,
    node_label_map=node_lab_map,
    chain_selection_map=chain_selection_map,
    graphein_config=gp.ProteinGraphConfig()
)