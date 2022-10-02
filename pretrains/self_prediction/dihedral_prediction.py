import torch 
import torch.nn as nn 
import torch.nn.functional as F

import numpy as np 
import time 

from pretrains.self_prediction.utils import get_bin, dihedral, dihedral_torch, NBIN

class DihedralPrediction(nn.Module):
    """
    Dihedral prediction 
    
    """ 
    def __init__(self, encoder, num_samples=512): 
        super().__init__()
        self.encoder = encoder 
        self.num_samples = num_samples
        self.mlp_heads = nn.Sequential(nn.Linear(self.encoder.hiddens[-1]*4, self.encoder.hiddens[-1]), nn.Dropout(0.25), nn.Linear(self.encoder.hiddens[-1], NBIN))
    
    def forward(self, data):
        t = time.time()
        total_samples = self.get_dihedral_samples(data)
        mask, masked_data, masked_samples = self.sample_data(data, total_samples)
        masked_label = self.get_masked_label(masked_samples, data.coords_tensor)
        x = self.encoder(masked_data)
        x0 = x[masked_samples[:,0]]
        x1 = x[masked_samples[:,1]]
        x2 = x[masked_samples[:,2]]
        x3 = x[masked_samples[:,3]]
        x = torch.cat([x0,x1,x2,x3],dim=-1)
        x = self.mlp_heads(x) # fangle
        loss = self.compute_loss(x, masked_label) 
        return loss

    def compute_loss(self, x, y):
        x = F.log_softmax(x,dim=-1)
        loss_fn = nn.NLLLoss()
        return loss_fn(x,y)

    def get_masked_label(self, masked_samples, coords_tensor): 
        ang = dihedral_torch(coords_tensor[masked_samples[:,0]], coords_tensor[masked_samples[:,1]], coords_tensor[masked_samples[:,2]], coords_tensor[masked_samples[:,3]])
        masked_label = get_bin(ang)
        return masked_label 

    def sample_data(self, data, total_samples): 
        # MASK_INDICATOR = -1
        num_total_samples = len(total_samples)
        idx = np.random.choice(num_total_samples, self.num_samples)
        mask = torch.zeros(num_total_samples).to(torch.bool) 
        mask[idx] = True # to mask out 
        masked_samples = total_samples[mask,: ]
        masked_data = data  # TODO do we need to mask out some attribute? 
        return mask, masked_data, masked_samples
        
    def get_dihedral_samples(self,data): # 0.013s/1forward
        samples = []
        from torch_geometric.utils import degree, to_dense_adj
        from pretrains.self_prediction.utils import unique
        deg_thr = 3 # IMPORTANT
        for e_type_idx in (range(data.kind.shape[1])):
            mask = data.kind[:,e_type_idx]==1
            edge_index = data.edge_index[:,mask]
            deg = degree(edge_index[0], num_nodes = data.num_nodes)
            mask = deg > deg_thr
          
            if mask.any():
                edge_mask = mask[edge_index[0,:]] # masking out relevant edges
                edge_index = edge_index[:,edge_mask] 
                edge_index = torch.unique(edge_index, dim=1)
                edge_index,_ = torch.sort(edge_index, dim=0)

                # get neighbor of a given center 
                center = edge_index[0,:]
                mask = torch.ones(len(center)).to(torch.bool)
                unique_center,first_indices = unique(center, dim=0) # neighbor0
                tmp = edge_index[:, first_indices][0]
                tmp0 = edge_index[:, first_indices] 
                mask[first_indices] = 0 
                
                edge_index = edge_index[:, mask]

                center = edge_index[0,:] 
                mask = torch.ones(len(center)).to(torch.bool)
                unique_center,first_indices = unique(center, dim=0) # neighbor1
                tmp1 = edge_index[:, first_indices] 
                mask[first_indices] = 0 

                edge_index = edge_index[:, mask]

                center = edge_index[0,:] 
                mask = torch.ones(len(center)).to(torch.bool)
                unique_center,first_indices = unique(center, dim=0) # neighbor2
                tmp2 = edge_index[:, first_indices] 
                mask[first_indices] = 0 

                # assert (tmp0[0]==tmp1[0]).all() 
                tmp0 = tmp0[1]
                tmp1 = tmp1[1]
                tmp2 = tmp2[1]

                samples_e_type_idx = torch.cat([tmp.unsqueeze(0),tmp0.unsqueeze(0),tmp1.unsqueeze(0),tmp2.unsqueeze(0)],axis=0).t()
                samples.append(samples_e_type_idx)
                del tmp
                del tmp0
                del tmp1 
                del tmp2 
                del samples_e_type_idx
            del mask 
            del deg 
            del edge_index 
            
        samples = torch.cat(samples,dim=0) #.to(data.node_feat.device)
        samples = torch.unique(samples, dim=0)
        return samples

    # deprecated
    # def __get_dihedral_samples(self, data): 
    #     from torch_geometric.utils import degree, to_dense_adj
    #     deg = degree(data.edge_index[0])
    #     dense_adj = to_dense_adj(data.edge_index)[0]
    #     samples = []
    #     for i in range(data.node_feat.shape[0]): 
    #         if deg[i]>3: 
    #             # n_i = torch.argwhere(dense_adj[i]==1)
    #             n_i = torch.nonzero(dense_adj[i], as_tuple=True)[0].cpu().tolist()
    #             i0 = n_i[0]
    #             i1 = n_i[1]  # TODO sampling? 
    #             i2 = n_i[2]
    #             i3 = n_i[3]
    #             samples.append([i0, i1, i2, i3])
    #     del dense_adj
    #     return torch.tensor(samples).to(data.node_feat.device)
