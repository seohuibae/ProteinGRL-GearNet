# Reaction Classification (Hermosilla et al. (2021))
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file ProtFunctdataSet.py
    \brief Dataset for the task of enzyme classification.
    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.
    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import sys
import os
import h5py
import numpy as np
import copy
import warnings

from graphein.protein import construct_graph, construct_graphs_mp
from config.protein import graphein_config, graph_format_convertor
from graphein.ml import ProteinGraphDataset, InMemoryProteinGraphDataset


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASKS_DIR = os.path.dirname(BASE_DIR)
ROOT_PROJ_DIR = os.path.dirname(TASKS_DIR)
sys.path.append(ROOT_PROJ_DIR)

def load_RX_dataset (pDataset = "training", root="../data-downstream", num_cores=16, run_process=False, transform_pyg=False): 
    pPath = root + "/ProtFunct"

    # Save the dataset path.
    path_ = pPath

    # Load the file with the list of functions.
    functions_ = []
    with open(pPath+"/unique_functions.txt", 'r') as mFile:
        for line in mFile:
            functions_.append(line.rstrip())

    proteinNames_ = [] # data
    with open(pPath+"/"+pDataset+".txt", 'r') as mFile:
        for line in mFile:
            proteinNames_.append(line.rstrip())

    # Load the functions. # label
    print("Reading protein functions")
    protFunct_ = {}
    with open(pPath+"/chain_functions.txt", 'r') as mFile:
        for line in mFile:
            splitLine = line.rstrip().split(',')
            if splitLine[0] in proteinNames_: 
                protFunct_[splitLine[0]] = int(splitLine[1])
    
    # Load the dataset
    print("Reading the data")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graphCache = {}
        pdb_codes_ = []
        chain_selections_ = [] 
        graph_labels_ = []
        cnt = 0
        for fileIter, pname in enumerate(proteinNames_):
            curClass = protFunct_[proteinNames_[fileIter]]
            # fileName = curFile.split('/')[-1]
            # if fileIter%250 == 0:
            #     print("\r# Reading file "+fileName+" ("+str(fileIter)+" of "+\
            #         str(len(fileList_))+")", end="")
            pname = pname.split('.')
            pdb_code, chain = pname[0], pname[1]
            pdb_codes_.append(pdb_code)
            chain_selections_.append(chain)
            graph_labels_.append(curClass)
            
    
    print(f"{pDataset}: {cnt}/{fileIter+1} not loaded ")

    # dataset = ProteinGraphDataset(root=pPath, pdb_codes=pdb_codes_, chain_selections=chain_selections_, graph_labels=graph_labels_, graphein_config=graphein_config, graph_format_convertor=graph_format_convertor, num_cores=num_cores, run_process=run_process)        
    process_in_separate_dir = pDataset
    if run_process:
        dataset = ProteinGraphDataset(root=pPath, process_in_separate_dir=process_in_separate_dir, pdb_codes=pdb_codes_, chain_selections=chain_selections_, graph_labels=graph_labels_, graphein_config=graphein_config, graph_format_convertor=graph_format_convertor, num_cores=num_cores, run_process=True, transform_pyg=transform_pyg)
        np.save(open(pPath+"/"+pDataset+"-idx-processed.npy",'wb'), np.array(dataset.get_idx_processed()))
        print(dataset.get_idx_processed())
        print('check data')
        print(dataset[0])
        print('done')
    else:
        idx_processed = np.load(open(pPath+"/"+pDataset+"-idx-processed.npy",'rb')).tolist()
        pdb_codes_=[pdb_codes_[i] for i in idx_processed]
        chain_selections_=[chain_selections_[i] for i in idx_processed]
        graph_labels_=[graph_labels_[i] for i in idx_processed]
        dataset = ProteinGraphDataset(root=pPath, process_in_separate_dir=process_in_separate_dir, pdb_codes=pdb_codes_, chain_selections=chain_selections_, graph_labels=graph_labels_, graphein_config=graphein_config, graph_format_convertor=graph_format_convertor, num_cores=num_cores, run_process=False, transform_pyg = transform_pyg)
        print('check data')
        print(dataset[0])
        print('done')
    return dataset