import torch 
import torch.nn as nn 
import torch.nn.functional as F

import numpy as np

from pretrains.self_prediction.utils import get_bin, cosinus, cosinus_torch, NBIN

class AnglePrediction(nn.Module):
    """
    Angle prediction 
    """ 
    def __init__(self, encoder, num_samples=512): 
        super().__init__()
        self.encoder = encoder 
        self.num_samples = num_samples
        self.mlp_heads = nn.Sequential(nn.Linear(self.encoder.hiddens[-1]*3, self.encoder.hiddens[-1]), nn.Dropout(0.25), nn.Linear(self.encoder.hiddens[-1], NBIN))
    
    def forward(self, data):
        total_samples = self.get_angle_samples(data)
        mask, masked_data, masked_samples = self.sample_data(data, total_samples) 
        masked_label = self.get_masked_label(masked_samples, data.coords_tensor)
        x = self.encoder(masked_data)
        x0 = x[masked_samples[:,0]]
        x1 = x[masked_samples[:,1]]
        x2 = x[masked_samples[:,2]]
        x = torch.cat([x0,x1,x2], dim=-1)
        x = self.mlp_heads(x) # fangle
        loss = self.compute_loss(x, masked_label) 
        return loss

    def compute_loss(self, x, y):
        x = F.log_softmax(x,dim=-1)
        loss_fn = nn.NLLLoss()
        return loss_fn(x,y)

    def get_masked_label(self, masked_samples, coords_tensor): 
        ang = cosinus_torch(coords_tensor[masked_samples[:,0]], coords_tensor[masked_samples[:,1]], coords_tensor[masked_samples[:,2]])
        masked_label = get_bin(ang)
        return masked_label 

    def sample_data(self, data, total_samples): 
        # MASK_INDICATOR = -1
        num_total_samples = len(total_samples)
        idx = np.random.choice(num_total_samples, self.num_samples)
        mask = torch.zeros(num_total_samples).to(torch.bool) 
        mask[idx] = True # to mask out 
        masked_samples = total_samples[mask,:]
        masked_data = data  # TODO do we need to mask out some attribute? 
        return mask, masked_data, masked_samples

    def get_angle_samples(self,data): # directed? undirected?
        from torch_geometric.utils import degree, to_dense_adj
        deg = degree(data.edge_index[0])
        dense_adj = to_dense_adj(data.edge_index)[0]
        samples = []
        for i in range(data.node_feat.shape[0]): 
            if deg[i]>2: 
                # n_i = torch.argwhere(dense_adj[i]==1)
                n_i = torch.nonzero(dense_adj[i], as_tuple=True)[0].cpu().tolist()
                # print(n_i, i)
                i0 = n_i[0] # self-loop
                i1 = n_i[1]  # TODO sampling? 
                i2 = n_i[2]
                samples.append([i0, i1, i2])
        return torch.tensor(samples).to(data.node_feat.device)
