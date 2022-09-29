from typing import Any, Callable, Iterable, List, Optional
import torch 
import numpy as np 

def annotate_node_metadata_pyg(data, funcs: List[Callable]): 
    for func in funcs:
        data = func(data)
    return data

def annotate_edge_metadata_pyg(data, funcs: List[Callable]): 
    for func in funcs:
        data = func(data)
    return data

#######################################################
def edge_feature_gearnet_pyg(data):
    assert data.node_feat is not None
    from utils.data.protein import EDGE_TYPE_GEARNET
    from utils.data.protein import onek_encoding_unk_list
    # def parse_sequence_dist(node_id, edge_index_tensor): 
    #     edges = edge_index_tensor.t().detach().cpu().tolist()
    #     seq_dist = []
    #     for u,v in edges: 
    #         dist = abs(int(node_id[v].split(':')[-1])-int(node_id[u].split(':')[-1]))
    #         seq_dist.append(dist) 
    #     return seq_dist
    node_feature_u = data.node_feat[data.edge_index[0]]
    node_feature_v = data.node_feat[data.edge_index[1]]
    spatial_dist = torch.tensor([data.distmat[e[0], e[1]] for e in data.edge_index.t()]).to(data.node_feat.device).unsqueeze(1)
    sequence_dist = torch.abs(data.edge_index[0]-data.edge_index[1]).unsqueeze(1) # node ids are sorted in order 
    edge_one_hot = data.kind 
    # sequence_dist = torch.FloatTensor(parse_sequence_dist(data.node_id, data.edge_index)).unsqueeze(1)     
    # edge_one_hot = torch.LongTensor(onek_encoding_unk_list(data.kind, allowable_set=EDGE_TYPE_GEARNET)) # TODO nodewise
    features = torch.cat([node_feature_u, node_feature_v, edge_one_hot, sequence_dist, spatial_dist], axis=1)
    data.edge_feat = features
    return data 

def edge_feature_ieconv_pyg(data):
    """
    Instead of intrinsic and extrinsic distances in
    the original IEConv layer, we follow New IEConv, which adopts three relative positional features
    proposed in Ingraham et al. (2019) and further augments them with additional input functions.
    the structural encodings e(s)ij with the positional encodings e(p)ij and then linearly transforming them tohave the same dimension as the model
    """
    raise NotImplementedError


def construct_relational_graph_among_edges(data): ## TODO TOO LONG
    from utils.data.protein import EDGE_TYPE_GEARNET
    from pretrains.self_prediction.utils import cosinus_torch, get_bin, NBIN
    new_num_nodes = len(data.edge_index[0])
    new_node_feat = data.edge_feat # change 
    device = new_node_feat.device
    node_triplets = []
    new_edge_index = []
    new_edge_type = []
    for u,(i,j) in enumerate(data.edge_index.t()): 
        for v,(w,k) in enumerate(data.edge_index.t()): 
            if j==w and i!=k:
                new_edge_index.append([u,v])
                ang = cosinus_torch(data.coords_tensor[i].unsqueeze(0), data.coords_tensor[j].unsqueeze(0), data.coords_tensor[k].unsqueeze(0))
                type = get_bin(ang) # 8bins 
                new_edge_type.append(type)
    data.edge_index = torch.LongTensor(new_edge_index).t().to(device)
    data.node_feat = new_node_feat 
    data.num_nodes = new_num_nodes
    data.kind = torch.LongTensor(new_edge_type).to(device)
    print(data)
    return data 

#######################################################
def annotate_distmat_pyg(data):
    dists_list = compute_distmat_pyg(data)
    distmat = torch.zeros(data.num_nodes, data.num_nodes).to(data.node_feat.device) # TODO too big? 
    lidx=0
    for i in range(len(dists_list)):
        ridx = lidx+dists_list[i].shape[0]
        distmat[lidx:ridx, lidx:ridx] = dists_list[i]
        lidx = ridx
    data.distmat = distmat
    return data

def compute_distmat_pyg(data):
    assert data.coords is not None 
    import scipy 
    eucl_dists_list = [] 
    nidx = torch.tensor([i for i in range(data.num_nodes)])
    for i in range(len(data.coords)):
        eucl_dists = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(data.coords[i],p=2))
        eucl_dists = torch.FloatTensor(eucl_dists)
        eucl_dists_list.append(eucl_dists)    
    return eucl_dists_list
    



