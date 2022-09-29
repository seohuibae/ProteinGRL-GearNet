import glob 
from config.protein import params_to_change, graphein_config, columns, graph_format_convertor
from graphein.ml import ProteinGraphDataset, InMemoryProteinGraphDataset

def load_swissprot_dataset (root, num_cores=16, run_process=False, transform_pyg=False): 
    pPath = root+'/swissprot/'
    pdb_paths = glob.glob(pPath+"pdb/*")
    dataset = ProteinGraphDataset(root=pPath, pdb_paths=pdb_paths, graphein_config=graphein_config, graph_format_convertor=graph_format_convertor, num_cores=num_cores, ext='gz', run_process=run_process, transform_pyg=transform_pyg) 
    # if run_process:
    #     dataset = ProteinGraphDataset(root=root, pdb_paths=pdb_paths, graphein_config=graphein_config, graph_format_convertor=graph_format_convertor, num_cores=num_cores, ext='gz', run_process=True, transform_pyg=transform_pyg)
    #     # np.save(open(pPath+"/"+pDataset+"-idx-processed.npy",'wb'), np.array(dataset.get_idx_processed()))
    #     # print(dataset.get_idx_processed())
    #     # print('check data')
    #     # print(dataset[0])
    #     # print('done')
    # else:
    #     idx_processed = np.load(open(pPath+"/"+pDataset+"-idx-processed.npy",'rb')).tolist()
    #     pdb_codes_=[pdb_codes_[i] for i in idx_processed]
    #     chain_selections_=[chain_selections_[i] for i in idx_processed]
    #     graph_labels_=[graph_labels_[i] for i in idx_processed]
    #     dataset = ProteinGraphDataset(root=root, pdb_paths=pdb_paths, graphein_config=graphein_config, graph_format_convertor=graph_format_convertor, num_cores=num_cores, ext='gz', run_process=False, transform_pyg = transform_pyg)
    #     print('check data')
    #     print(dataset[0])
    #     print('done')
    return dataset