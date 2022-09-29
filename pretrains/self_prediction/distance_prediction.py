import torch 
import torch.nn as nn 

import numpy as np 

class DistancePrediction(nn.Module):
    """
    distance prediction 
    
    """ 
    def __init__(self, encoder, num_samples=512): 
        super().__init__()
        self.encoder = encoder 
        self.num_samples = num_samples
        self.mlp_heads = nn.Sequential(nn.Linear(self.encoder.hiddens[-1]*2, self.encoder.hiddens[-1]), nn.Dropout(0.25), nn.Linear(self.encoder.hiddens[-1],1))
    
    def forward(self, data):
        mask, masked_data = self.sample_data(data) 
        masked_label = self.get_masked_label(data, mask)
        x = self.encoder(masked_data)
        xi = x[data.edge_index[0][mask]]
        xj = x[data.edge_index[1][mask]]
        x = torch.cat([xi,xj],dim=-1)
        x = self.mlp_heads(x).squeeze(1) # fdist
        loss = self.compute_loss(x, masked_label) 
        return loss

    def compute_loss(self, x, y): 
        loss_fn = nn.MSELoss()
        return loss_fn(x,y)
    
    def get_masked_label(self, data, mask): # .dist_mat 
        pdist = nn.PairwiseDistance(p=2)
        x1 = data.coords_tensor[data.edge_index[0][mask]]
        x2 = data.coords_tensor[data.edge_index[1][mask]]
        label = pdist(x1, x2).to(torch.float)
        return label 

    def sample_data(self, data): 
        # MASK_INDICATOR = -1
        num_edges = data.edge_index.shape[1]
        idx = np.random.choice(num_edges, self.num_samples)
        mask = torch.zeros(num_edges).to(torch.bool) 
        mask[idx] = True # to mask out 
        
        # masked_edge_index = data.edge_index
        # masked_kind = data.kind 
        # # TODO else? 
        # data.edge_index = masked_edge_index[:,~mask]
        # data.kind = masked_kind[:,~kind]
        return mask, data 

        