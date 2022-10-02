import torch
import torch.nn as nn 

import numpy as np 
import random 
import copy 

from models.layers import pooling_layer

class MultiviewContrast(nn.Module): 
    def __init__(self, encoder, pooling, batch_size, len_subsequence, radius_subspace, edge_mask_p=0.15, temperature=0.07): 
        super().__init__()
        self.encoder = encoder 
        self.batch_size = batch_size 

        self.augmentation_dict = {
        'aug1': (lambda x: random_edge_masking(edge_mask_p)(subsequence_cropping_fast(batch_size,len_subsequence)(x))),
        'aug2': (lambda x: identity(subspace_cropping_fast(batch_size,radius_subspace)(x)))}
        
        self.aug1 = self.augmentation_dict['aug1'] 
        self.aug2 = self.augmentation_dict['aug2']
        self.pooling = pooling
        self.mlp_heads = nn.Sequential(nn.Linear(self.encoder.hiddens[-1], self.encoder.hiddens[-1]), nn.Dropout(0.25), nn.Linear(self.encoder.hiddens[-1],self.encoder.hiddens[-1]))
        self.loss = InfoNCELoss(batch_size, temperature)
    
    def forward(self, data):
        data = self.process_batch(data) 
        data2 = copy.deepcopy(data)
        data = self.aug1(data)
        data2 = self.aug2(data2) 
        zx = self.pooling(self.encoder(data), data.batch)
        zy = self.pooling(self.encoder(data2), data2.batch)
        zx = self.mlp_heads(zx)
        zy = self.mlp_heads(zy)
        pt_loss = self.loss(zx, zy) 
        return pt_loss
    
    def process_batch(self, data):
        ends_batch = []
        for b in range(self.batch_size):
            lst = (data.batch==b).nonzero(as_tuple=True)[0].cpu().tolist()
            ends_batch.append([lst[0], lst[-1]])
        ends_batch = torch.tensor(ends_batch)
        data.ends = ends_batch
        return data 


class InfoNCELoss(nn.Module):
    # https://medium.com/the-owl/simclr-in-pytorch-5f290cb11dd7
    def __init__(self, batch_size, temperature):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    # loss 분모 부분의 negative sample 간의 내적 합만을 가져오기 위한 마스킹 행렬
    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        # loss 분자 부분의 원본 - augmentation 이미지 간의 내적 합을 가져오기 위한 부분
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        
        labels = torch.from_numpy(np.array([0]*N)).reshape(-1).to(positive_samples.device).long()
        
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        
        return loss

####### augs 

def identity(data):
    return data 

def random_edge_masking(edge_mask_p):
    def fn(data):
        mask = (torch.rand(len(data.edge_index[0])) > edge_mask_p)
        data.edge_index = data.edge_index[:,mask]
        data.kind = data.kind[mask]
        return data  
    return fn

def subsequence_cropping(batch_size, len_subsequence): # 
    def fn(data): 
        ends = data.ends.cpu().tolist()
        sampled_subset_batch = []
        for b in range(batch_size):
            lidx, ridx = ends[b][0], ends[b][1]
            lst = torch.arange(lidx, ridx+1)
            # idx = random.sample(range(lidx, ridx+1), 1)[0]   
            idx = lst[torch.randperm(ridx-lidx+1)[0]]  
            while (idx+len_subsequence)>ridx: 
                # idx = random.sample(range(lidx, ridx+1), 1)[0]
                idx = lst[torch.randperm(ridx-lidx+1)[0]]  
            subset = torch.arange(idx,idx+len_subsequence)
            sampled_subset_batch.append(subset)
        sampled_subset_batch = torch.cat(sampled_subset_batch)
        
        sampled_node_feat, sampled_edge_index, sampled_batch, sampled_kind = subgraph(sampled_subset_batch,data.node_feat, data.edge_index, data.batch, data.num_nodes, edge_attr= data.kind, relabel_nodes=True)
        data.node_feat = sampled_node_feat
        data.num_nodes = len(sampled_node_feat)
        data.edge_index = sampled_edge_index 
        data.batch = sampled_batch
        data.kind = sampled_kind
        return data 
    return fn

def subspace_cropping(batch_size, radius_subspace): # radius d: 15
    pdist = nn.PairwiseDistance(p=2)
    def fn(data):
        ends = data.ends.cpu().tolist()
        sampled_subset_batch = [] # p 
        for b in range(batch_size):
            lidx, ridx = ends[b][0], ends[b][1]
            lst = torch.arange(lidx, ridx+1)
            # idx = random.sample(range(lidx, ridx+1), 1)[0]      # get residue p    
            idx = lst[torch.randperm(ridx-lidx+1)[0]]
            distance = pdist(data.coords_tensor[idx,:].unsqueeze(0).repeat(ridx-lidx+1,1), data.coords_tensor[lidx:ridx+1,:])
            subset = torch.arange(lidx,ridx+1)[distance<radius_subspace]
            sampled_subset_batch.append(subset)
        sampled_subset_batch = torch.cat(sampled_subset_batch)
        sampled_node_feat, sampled_edge_index, sampled_batch, sampled_kind = subgraph(sampled_subset_batch, data.node_feat, data.edge_index, data.batch, data.num_nodes,edge_attr= data.kind, relabel_nodes=True)
        data.node_feat = sampled_node_feat
        data.num_nodes = len(sampled_node_feat)
        data.edge_index = sampled_edge_index 
        data.batch = sampled_batch
        data.kind = sampled_kind
        return data 
    return fn 
    
def subsequence_cropping_fast(batch_size, len_subsequence): # 
    def fn(data): 
        device = data.node_feat.device 
        lidx, ridx = data.ends[:,0].cpu().tolist(), data.ends[:,1].cpu().tolist() 
        mask = data.ends[:,1]-data.ends[:,0] > len_subsequence 
        idx = np.random.randint(lidx, [r-len_subsequence+1 if mask[b] else r for b,r in enumerate(ridx)])
        sampled_subset_batch = torch.cat([torch.arange(i, i+len_subsequence) if mask[b] else torch.arange(lidx[b], ridx[b]+1) for b,i in enumerate(idx)]).to(device)    
        sampled_node_feat, sampled_edge_index, sampled_batch, sampled_kind = subgraph(sampled_subset_batch,data.node_feat, data.edge_index, data.batch, data.num_nodes, edge_attr= data.kind, relabel_nodes=True)
        data.node_feat = sampled_node_feat
        data.num_nodes = len(sampled_node_feat)
        data.edge_index = sampled_edge_index 
        data.batch = sampled_batch
        data.kind = sampled_kind
        return data 
    return fn

def subspace_cropping_fast(batch_size, radius_subspace): # radius d: 15
    pdist = nn.PairwiseDistance(p=2)
    def fn(data):
        device = data.node_feat.device 
        lidx, ridx = data.ends[:,0], data.ends[:,1] 
        idx = np.random.randint(lidx.cpu().tolist(), [r+1 for r in ridx.cpu().tolist()])
        sampled_subset_batch = [] # p 
        for b in range(batch_size):
            dist = pdist(data.coords_tensor[idx[b],:].unsqueeze(0).repeat(ridx[b]-lidx[b]+1,1), data.coords_tensor[lidx[b]:ridx[b]+1,:])
            subset = torch.arange(lidx[b],ridx[b]+1)[dist<radius_subspace]
            sampled_subset_batch.append(subset)
        sampled_subset_batch = torch.cat(sampled_subset_batch).to(device)
        sampled_node_feat, sampled_edge_index, sampled_batch, sampled_kind = subgraph(sampled_subset_batch, data.node_feat, data.edge_index, data.batch, data.num_nodes,edge_attr= data.kind, relabel_nodes=True)
        data.node_feat = sampled_node_feat
        data.num_nodes = len(sampled_node_feat)
        data.edge_index = sampled_edge_index 
        data.batch = sampled_batch
        data.kind = sampled_kind
        return data 
    return fn 


####### utils 
def index_to_mask(index, size=None):
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask

def subgraph(
    subset,
    node_feat,
    edge_index,
    batch,
    num_nodes,
    edge_attr = None,
    relabel_nodes = True,
    return_edge_mask = False):
    # from torch_geometric.utils import subgraph
    device = edge_index.device

    if isinstance(subset, (list, tuple)):
        subset = torch.tensor(subset, dtype=torch.long, device=device)

    # if subset.dtype == torch.bool or subset.dtype == torch.uint8:
    #     num_nodes = subset.size(0)
    # else:
        # num_nodes = maybe_num_nodes(edge_index, num_nodes)
    subset = index_to_mask(subset, size=num_nodes)

    node_mask = subset
    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    if relabel_nodes:
        node_idx = torch.zeros(node_mask.size(0), dtype=torch.long,
                               device=device)
        node_idx[subset] = torch.arange(subset.sum().item(), device=device)
        node_feat = node_feat[node_mask]
        batch = batch[node_mask]
        edge_index = node_idx[edge_index]
        return node_feat, edge_index, batch, edge_attr
    return node_feat, edge_index, batch, edge_attr

