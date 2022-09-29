import torch
import torch.nn as nn 
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_geometric.nn.inits import zeros
# from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes

from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

pooling_layer = {
    'mean': global_mean_pool,
    'max': global_max_pool, 
    'add': global_add_pool,
}

class GraphConvolutionLayer(nn.Module):
    """
    Graph convolution layers
    """
    def __init__(self, input_dim, output_dim, conv='gcn',
                dropout=0.,
                activation=nn.ReLU(), 
                bias=False,
                featureless=False, 
                add_self_loop=False, bn=False, skip=False, **kwargs):
        super(GraphConvolutionLayer, self).__init__(**kwargs)

        self.featureless = featureless
        self.input_dim = input_dim 
        self.output_dim = output_dim
        self.conv_name = conv 
        ############################
        if conv == 'gcn':
            from torch_geometric.nn.conv import GCNConv
            self.conv = GCNConv(input_dim, output_dim, bias=bias, add_self_loops=add_self_loop)
        elif conv == 'gat':
            from torch_geometric.nn.conv import GATConv
            self.conv = GATConv(input_dim, output_dim, bias=bias, add_self_loops=add_self_loop)
        elif conv == 'gin':
            from torch_geometric.nn.conv import GINConv 
            self.conv = GINConv(input_dim, output_dim, bias=bias, add_self_loops=add_self_loop)
        elif conv == 'ieconv':
            from models.convs import SimplifiedIEConv
            self.conv = SimplifiedIEConv(input_dim, output_dim, bias=bias) # TODO Aggregation for Simplified IEConv layer
        elif conv  == 'rgcn':
            from torch_geometric.nn.conv import FastRGCNConv 
            from utils.data.protein import EDGE_TYPE_GEARNET
            self.conv = FastRGCNConv(input_dim, output_dim, num_relations=len(EDGE_TYPE_GEARNET), aggr='add', root_weight=False, bias=bias)
        elif conv  == 'gearnet':
            from models.convs import GearNetConv
            self.conv = GearNetConv(input_dim, output_dim, bias=bias) # TODO Aggregation for Relational Graph Convolutional Layer (Schlichtkrull et al., 2018) 
        elif conv == 'gearnet-edge':
            from models.convs import GearNetEdgeConv
            self.conv = GearNetEdgeConv(input_dim, output_dim, bias=bias) # TODO Aggregation function for GearNet-Edge (Zhang et al., 2022)
        else:
            raise NotImplementedError
        ###########################
        if not bn: 
            self.bn = None 
        else: 
            self.bn = nn.BatchNorm1d(output_dim)
        self.activation = activation
        self.dropout = dropout
        self.skip = skip

    def forward(self, x, edge_index, training=None, **kwargs):
        # Convolution - Batch Normalization - Activation - Dropout - Pooling
        if self.skip:
            # assert x_in.shape[0] == self.conv[1]
            x_in = x.clone().detach() 
        # convolve
        if self.conv_name in ['gcn', 'gat', 'gin']:
            x = self.conv(x, edge_index)
        elif self.conv_name  in ['ieconv']:
            x = self.conv(x, edge_index, edge_weight = kwargs['ie_edge_feat'])
        elif self.conv_name  in ['gearnet','gearnet-ieconv']:
            x = self.conv(x, edge_index, edge_type=kwargs['edge_type'])
        elif self.conv_name  in ['gearnet-edge', 'gearnet-edge-ieconv']:
            x = self.conv(x, edge_index, edge_type=kwargs['edge_type'], message_jir=kwargs['edge_feat'])
        else:
            raise NotImplementedError
        # batch norm 
        if self.bn is not None: 
            x = self.bn(x)
        # activation
        x = self.activation(x)

        # skip connection
        if self.skip and x.shape[1] == x_in.shape[1]: 
            x += x_in 
        # dropout
        if isinstance(training, bool) and training:
            x = F.dropout(x, self.dropout)        
        return x


class EdgeMessagePassingLayer(GraphConvolutionLayer):
    """
     Each node in the graph corresponds to an edge in the original graph.
    """
    def __init__(self, input_dim, output_dim, conv='rgcn', 
                dropout=0.,
                activation=nn.ReLU(), 
                bias=False,
                featureless=False, 
                add_self_loop=False, bn=True, skip=False, **kwargs):
        super(EdgeMessagePassingLayer, self).__init__(input_dim, output_dim, conv,dropout, activation, 
                bias, featureless, add_self_loop, bn, skip, **kwargs)
        assert conv == 'rgcn'
        assert bn == True 
        assert skip == False 

