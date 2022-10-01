# Fold Classification (Hermosilla et al. (2021))
"""SCOPe1.75 (Hou et al. 2018; Hermosilla et al. 2021)
    """

import os
import h5py
import copy
import warnings
import numpy as np
import glob

from graphein.protein import construct_graph, construct_graphs_mp
from config.protein import graphein_config, graph_format_convertor
from graphein.ml import ProteinGraphDataset, InMemoryProteinGraphDataset


def parse_path_to_dname(str):
    str = str.split('/')[-1].split('.')[0]
    return str

def load_FC_dataset(pDataset = "training", root='../data-downstream', num_cores=16, run_process=False, transform_pyg=False):
    pPath = root+"/HomologyTAPE"

    # Load the file with the list of classes.
    maxIndex = 0
    classes_ = {}
    with open(pPath+"/class_map.txt", 'r') as mFile:
        for line in mFile:
            lineList = line.rstrip().split('\t')
            classes_[lineList[0]] = int(lineList[1])
            maxIndex = max(maxIndex, int(lineList[1]))
    classesList_ = ["" for i in range(maxIndex+1)]
    for key, value in classes_.items():
        classesList_[value] = key

    # Get the file list.
    numProtsXCat = np.full((len(classes_)), 0, dtype=np.int32)
    fileList_ = []
    cathegories_ = []
    with open(pPath+"/"+pDataset+".txt", 'r') as mFile:
        for curLine in mFile:
            splitLine = curLine.rstrip().split('\t')
            curClass = classes_[splitLine[-1]] # class label
            fileList_.append(pPath+"/"+pDataset+"/"+splitLine[0]) # domain na,e
            numProtsXCat[curClass] += 1
            cathegories_.append(curClass) # graph label

    # Get paths for all pdbstyle-1.75
    pdb_paths_all = glob.glob(pPath+'/pdbstyle-1.75/*/*/**.ent', recursive=True)
    dnames_all = [parse_path_to_dname(path) for path in pdb_paths_all]
    dnames2paths = dict(zip(dnames_all, pdb_paths_all))
    # Load the dataset.
    onlyCAProts_ = set()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graphCache = {}
        pdb_paths_ = []
        chain_selections_ = []
        graph_labels_ = []
        cnt = 0
        for fileIter, (curFile, curClass) in enumerate(zip(fileList_, cathegories_)):
            className = curFile.split('/')[-2]
            while len(className) < 6:
                className = className+" "
            fileName = curFile.split('/')[-1]
            # if fileIter%100 == 0:
            #     print("\r# Reading file "+fileName+" / "+className+" ("+str(fileIter)+" of "+\
            #         str(len(fileList_))+")", end="")
            try:
                pdb_path = dnames2paths[fileName]
                chain = fileName[-2].upper()
                pdb_paths_.append(pdb_path)
                chain_selections_.append(chain)
                graph_labels_.append(curClass)
            except:
                cnt+=1
    print(f"{pDataset}: {cnt}/{fileIter+1} not available in SCOPEe1.75")

    # process_in_separate_dir = None
    # if pDataset in ["test_fold", "test_superfamily", "test_family"]:
    process_in_separate_dir = pDataset

    if run_process:
        dataset = ProteinGraphDataset(root=pPath, process_in_separate_dir=process_in_separate_dir, pdb_paths=pdb_paths_, chain_selections=chain_selections_, graph_labels=graph_labels_, graphein_config=graphein_config, graph_format_convertor=graph_format_convertor, num_cores=num_cores, ext='ent', dnames2paths=dnames2paths, run_process=True, transform_pyg = transform_pyg)
        np.save(open(pPath+"/"+pDataset+"-idx-processed.npy",'wb'), np.array(dataset.get_idx_processed()))
    else:
        idx_processed = np.load(open(pPath+"/"+pDataset+"-idx-processed.npy",'rb')).tolist()
        pdb_paths_=[pdb_paths_[i] for i in idx_processed]
        chain_selections_=[chain_selections_[i] for i in idx_processed]
        graph_labels_=[graph_labels_[i] for i in idx_processed]
        dataset = ProteinGraphDataset(root=pPath, process_in_separate_dir=process_in_separate_dir, pdb_paths=pdb_paths_, chain_selections=chain_selections_, graph_labels=graph_labels_, graphein_config=graphein_config, graph_format_convertor=graph_format_convertor, num_cores=num_cores, ext='ent', dnames2paths=dnames2paths, run_process=False, transform_pyg = transform_pyg)
    # pdb_paths_=pdb_paths_[:5]
    # chain_selections_=chain_selections_[:5]
    # graph_labels_=graph_labels_[:5]

    # if run_process:
    #     dataset = InMemoryProteinGraphDataset(root=pPath, process_in_separate_dir=process_in_separate_dir, name=process_in_separate_dir, pdb_paths=pdb_paths_, chain_selections=chain_selections_, graph_labels=graph_labels_, graphein_config=graphein_config, graph_format_convertor=graph_format_convertor, num_cores=num_cores, ext='ent', dnames2paths=dnames2paths, run_process=True, transform_pyg = transform_pyg)
    #     np.save(open(pPath+"/"+pDataset+"-idx-processed.npy",'wb'), np.array(dataset.get_idx_processed()))
    # else:
    #     idx_processed = np.load(open(pPath+"/"+pDataset+"-idx-processed.npy",'rb')).tolist()
    #     pdb_paths_=[pdb_paths_[i] for i in idx_processed]
    #     chain_selections_=[chain_selections_[i] for i in idx_processed]
    #     graph_labels_=[graph_labels_[i] for i in idx_processed]
    #     dataset = InMemoryProteinGraphDataset(root=pPath, process_in_separate_dir=process_in_separate_dir, name=process_in_separate_dir, pdb_paths=pdb_paths_, chain_selections=chain_selections_, graph_labels=graph_labels_, graphein_config=graphein_config, graph_format_convertor=graph_format_convertor, num_cores=num_cores, ext='ent', dnames2paths=dnames2paths, run_process=False, transform_pyg = transform_pyg)
    
    return dataset
