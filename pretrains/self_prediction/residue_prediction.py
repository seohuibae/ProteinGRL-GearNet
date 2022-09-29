import torch 
import torch.nn as nn 
import torch.nn.functional as F

import numpy as np 

from graphein.protein.resi_atoms import BASE_AMINO_ACIDS
from utils.data.protein import onek_encoding_unk_additional_list

class ResiduePrediction(nn.Module):
    """
    residue type prediction 
    randomly mask node features of some residues and then predict these masked residue types via structure-based encoders
    """ 
    def __init__(self, encoder, output_dim=21, num_samples=512): 
        super().__init__()
        self.encoder = encoder 
        self.output_dim = output_dim 
        self.num_samples = num_samples
        self.mlp_heads = nn.Sequential(nn.Linear(self.encoder.hiddens[-1], self.encoder.hiddens[-1]), nn.Dropout(0.25), nn.Linear(self.encoder.hiddens[-1], output_dim))
    
    def forward(self, data):
        mask, masked_data = self.sample_data(data) 
        masked_label = self.get_masked_label(data, mask)
        x = self.encoder(masked_data)
        x = self.mlp_heads(x)
        loss = self.compute_loss(x[mask,:], masked_label) 
        return loss

    def compute_loss(self, x, y): 
        x = F.log_softmax(x,dim=-1)
        loss_fn = nn.NLLLoss()
        return loss_fn(x,y)
    
    def get_masked_label(self, data, mask): # .residue_name 
        # label = onek_encoding_unk_additional_list(data.residue_name, BASE_AMINO_ACIDS)
        label = torch.argmax(data.node_feat, dim=1) # one hot encoded residue_name is already attributed as node_feat (dim 21)
        label = label[mask]
        return label 

    def sample_data(self, data): 
        MASK_INDICATOR = -1
        num_nodes = data.node_feat.shape[0]
        idx = np.random.choice(num_nodes, self.num_samples)
        mask = torch.zeros(num_nodes).to(torch.bool) 
        mask[idx] = True # to mask out 
        masked_node_feat = data.node_feat 
        masked_node_feat[mask] = MASK_INDICATOR
        data.node_feat = masked_node_feat
        return mask, data 
    

