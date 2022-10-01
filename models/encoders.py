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

from models.layers import GraphConvolutionLayer, EdgeMessagePassingLayer, pooling_layer

######
# GNN-based encoders  
class GCN(nn.Module):
    def __init__(self, input_dim, hiddens, output_dim, dropout, add_self_loop=True, bn=True, **kwargs):
        super(GCN, self).__init__(**kwargs)
        
        self.hiddens = hiddens
        self.layers_ = nn.ModuleList([])
        layer0 = GraphConvolutionLayer(input_dim=input_dim,
                                  output_dim=hiddens[0], conv='gcn', dropout=dropout,
                                  activation=nn.ReLU(), add_self_loop=add_self_loop, bn=bn)
        self.layers_.append(layer0)
        nhiddens = len(hiddens)
        for _ in range(1,nhiddens):
            layertemp = GraphConvolutionLayer(input_dim=hiddens[_-1],
                                      output_dim=hiddens[_], conv='gcn', dropout=dropout,
                                      activation=nn.ReLU(), add_self_loop=add_self_loop, bn=bn)
            self.layers_.append(layertemp)

    def forward(self, data, training=None):
        x, edge_index = data.node_feat.to(torch.float), data.edge_index
        for layer in self.layers_:
            x = layer(x, edge_index, training)
        return x 

class GAT(nn.Module):
    def __init__(self, input_dim, hiddens, output_dim, dropout, add_self_loop=True, bn=True, **kwargs):
        super(GAT, self).__init__(**kwargs)
        
        self.hiddens = hiddens
        self.layers_ = nn.ModuleList([])
        layer0 = GraphConvolutionLayer(input_dim=input_dim,
                                  output_dim=hiddens[0], conv='gat', dropout=dropout,
                                  activation=nn.ReLU(), add_self_loop=add_self_loop, bn=bn)
        self.layers_.append(layer0)
        nhiddens = len(hiddens)
        for _ in range(1,nhiddens):
            layertemp = GraphConvolutionLayer(input_dim=hiddens[_-1],
                                      output_dim=hiddens[_], conv='gat', dropout=dropout,
                                      activation=nn.ReLU(), add_self_loop=add_self_loop, bn=bn)
            self.layers_.append(layertemp)

    def forward(self, data, training=None):
        x, edge_index = data.node_feat.to(torch.float), data.edge_index
        for layer in self.layers_:
            x = layer(x, edge_index, training)
        return x 

class GIN(nn.Module):
    def __init__(self, input_dim, hiddens, output_dim, dropout, add_self_loop=True, bn=True, **kwargs):
        super(GIN, self).__init__(**kwargs)
        
        self.hiddens = hiddens
        self.layers_ = nn.ModuleList([])
        self.nn = nn.Sequential()
        eps = 0.0
        train_eps = False 
        layer0 = GraphConvolutionLayer(input_dim=input_dim,
                                  output_dim=hiddens[0], conv='gin', dropout=dropout,
                                  activation=nn.ReLU(), add_self_loop=add_self_loop, bn=bn)
        self.layers_.append(layer0)
        nhiddens = len(hiddens)
        for _ in range(1,nhiddens):
            layertemp = GraphConvolutionLayer(input_dim=hiddens[_-1],
                                      output_dim=hiddens[_], conv='gin', dropout=dropout,
                                      activation=nn.ReLU(), add_self_loop=add_self_loop, bn=bn)
            self.layers_.append(layertemp)

    def forward(self, data, training=None):
        x, edge_index = data.node_feat.to(torch.float), data.edge_index
        for layer in self.layers_:
            x = layer(x, edge_index, training)
        return x 
######
# proposed encoders  
class GearNet(nn.Module): 
    def __init__(self, input_dim, hiddens, output_dim, dropout, **kwargs):
        super(GearNet, self).__init__(**kwargs)
    
        self.hiddens = hiddens
        self.layers_ = nn.ModuleList([])

        layer0 = GraphConvolutionLayer(input_dim=input_dim,
                                  output_dim=hiddens[0], conv='gearnet', dropout=dropout,
                                  activation=nn.ReLU(), bn=True, skip=True) # TODO
        self.layers_.append(layer0)
        nhiddens = len(hiddens)
        for _ in range(1,nhiddens):
            layertemp = GraphConvolutionLayer(input_dim=hiddens[_-1],
                                      output_dim=hiddens[_], conv='gearnet', dropout=dropout,
                                      activation=nn.ReLU(), bn=True, skip=True)
            self.layers_.append(layertemp)

    def forward(self, data, training=None):
        x, edge_index = data.node_feat, data.edge_index
        data.edge_type = torch.argmax(data.kind,1)
        for layer in self.layers_:
            x = layer(x, edge_index, training, edge_type=data.edge_type)
        return x 

class GearNetIEConv(nn.Module): 
    def __init__(self, input_dim, hiddens, output_dim, dropout, **kwargs):
        super(GearNet, self).__init__(**kwargs)

        self.hiddens = hiddens
        self.layers_ = nn.ModuleList([])
        layer0 = GraphConvolutionLayer(input_dim=input_dim,
                                  output_dim=hiddens[0], conv='gearnet', dropout=dropout,
                                  activation=nn.ReLU(), bn=True, skip=True) # TODO
        self.layers_.append(layer0)
        nhiddens = len(hiddens)
        for _ in range(1,nhiddens):
            layertemp = GraphConvolutionLayer(input_dim=hiddens[_-1],
                                      output_dim=hiddens[_], conv='gearnet', dropout=dropout,
                                      activation=nn.ReLU(), bn=True, skip=True)
            self.layers_.append(layertemp)

        self.ielayers_ = nn.ModuleList([])
        layer0 = GraphConvolutionLayer(input_dim=input_dim,
                                  output_dim=hiddens[0], conv='ieconv', dropout=dropout,
                                  activation=lambda x: x, bn=False, skip=False) # TODO
        self.ielayers_.append(layer0)
        nhiddens = len(hiddens)
        for _ in range(1,nhiddens):
            layertemp = GraphConvolutionLayer(input_dim=hiddens[_-1],
                                      output_dim=hiddens[_], conv='ieconv', dropout=dropout,
                                      activation=lambda x: x, bn=False, skip=False)
            self.ielayers_.append(layertemp)
     

    def forward(self, data, training=None):
        x, edge_index = data.node_feat, data.edge_index
        from utils.data.protein import edge_feature_ieconv_pyg
        data = edge_feature_ieconv_pyg(data) # TODO 
        for layer, ielayer in zip(self.layers_, self.ielayers_):
            x = layer(x, edge_index, training, edge_type=data.edge_type) + ielayer(x, edge_index, training, ie_edge_type=data.ie_edge_type)
            
        return x 

class GearNetEdge(nn.Module): 
    def __init__(self, input_dim, hiddens, output_dim, dropout, **kwargs):
        super(GearNetEdge, self).__init__(**kwargs)
        
        self.hiddens = hiddens
        self.layers_ = nn.ModuleList([])
        layer0 = GraphConvolutionLayer(input_dim=input_dim,
                                  output_dim=hiddens[0], conv='gearnet-edge', dropout=dropout,
                                  activation=nn.ReLU(), bn=True, skip=True)
        self.layers_.append(layer0)
        nhiddens = len(hiddens)
        for _ in range(1,nhiddens):
            layertemp = GraphConvolutionLayer(input_dim=hiddens[_-1],
                                      output_dim=hiddens[_], conv='gearnet-edge', dropout=dropout,
                                      activation=nn.ReLU(),bn=True, skip=True)
            self.layers_.append(layertemp)

        self.mlayers_ = nn.ModuleList([])
        layer0 = EdgeMessagePassingLayer(input_dim=input_dim,output_dim=hiddens[0], dropout=dropout)
        self.mlayers_.append(layer0)
        nhiddens = len(hiddens)
        for _ in range(1,nhiddens):
            layertemp = EdgeMessagePassingLayer(input_dim=hiddens[_-1],
                                      output_dim=hiddens[_], dropout=dropout)
            self.mlayers_.append(layertemp)

    def forward(self, data, training=None):
        x, edge_index = data.node_feat, data.edge_index
        from utils.data.protein import edge_feature_gearnet_pyg, annotate_distmat_pyg
        from utils.data.protein import construct_relational_graph_among_edges
        data = annotate_distmat_pyg(data)
        print(data)
        data = edge_feature_gearnet_pyg(data) # TODO 
        print(data)
        rdata = construct_relational_graph_among_edges(data) # TODO 
        print(rdata)
        mjir = rdata.node_feat 
        for layer, mlayer in zip(self.layers_, self.mlayers_):
            mjir = mlayer(mjir, rdata.edge_index)
            x = layer(x, edge_index, training, edge_type=data.edge_type, message_jir=mjir)
        return x 

class GearNetEdgeIEConv(nn.Module): 
    def __init__(self, input_dim, hiddens, output_dim, dropout, **kwargs):
        super(GearNetEdge, self).__init__(**kwargs)
        
        self.hiddens = hiddens
        self.layers_ = nn.ModuleList([])
        layer0 = GraphConvolutionLayer(input_dim=input_dim,
                                  output_dim=hiddens[0], conv='gearnet-edge', dropout=dropout,
                                  activation=nn.ReLU(), bn=True, skip=True)
        self.layers_.append(layer0)
        nhiddens = len(hiddens)
        for _ in range(1,nhiddens):
            layertemp = GraphConvolutionLayer(input_dim=hiddens[_-1],
                                      output_dim=hiddens[_], conv='gearnet-edge', dropout=dropout,
                                      activation=nn.ReLU(), bn=True, skip=True)
            self.layers_.append(layertemp)

        self.mlayers_ = nn.ModuleList([])
        layer0 = EdgeMessagePassingLayer(input_dim=input_dim,output_dim=hiddens[0], dropout=dropout)
        self.mlayers_.append(layer0)
        nhiddens = len(hiddens)
        for _ in range(1,nhiddens):
            layertemp = EdgeMessagePassingLayer(input_dim=hiddens[_-1],
                                      output_dim=hiddens[_], dropout=dropout)
            self.mlayers_.append(layertemp)

        self.ielayers_ = nn.ModuleList([])
        layer0 = GraphConvolutionLayer(input_dim=input_dim,
                                  output_dim=hiddens[0], conv='ieconv', dropout=dropout,
                                  activation=lambda x: x, add_self_loop=False, bn=False, skip=False) # TODO
        self.ielayers_.append(layer0)
        nhiddens = len(hiddens)
        for _ in range(1,nhiddens):
            layertemp = GraphConvolutionLayer(input_dim=hiddens[_-1],
                                      output_dim=hiddens[_], conv='ieconv', dropout=dropout,
                                      activation=lambda x: x, add_self_loop=False, bn=False, skip=False)
            self.ielayers_.append(layertemp)

    def forward(self, data, training=None):
        x, edge_index = data.node_feat, data.edge_index
        from utils.data.protein import edge_feature_gearnet_pyg, annotate_distmat_pyg
        from utils.data.protein import edge_feature_ieconv_pyg
        from utils.data.protein import construct_relational_graph_among_edges
        data = annotate_distmat_pyg(data)
        data = edge_feature_gearnet_pyg(data) # TODO 
        data = edge_feature_ieconv_pyg(data) # TODO 
        rdata = construct_relational_graph_among_edges(data) #TODO
        mjir = rdata.node_feat
        for layer, mlayer, ielayer in zip(self.layers_, self.mlayers_, self.ielayers_):
            mjir = mlayer(mjir, rdata.edge_index)
            x = layer(x, edge_index, training, edge_type=data.edge_type, message_jir=mjir) + ielayer(x, edge_index, training, ie_edge_type=data.ie_edge_type)
        return x 

######## 
# transformer-based encoders    
class GraphTransformer(nn.Module): 
    def __init__(self, input_dim, hiddens, output_dim, dropout, **kwargs):
        use_edge_feat = kwargs['use_edge_feat']
        del kwargs['use_edge_feat']
        super(GraphTransformer, self).__init__(**kwargs)
        self.use_edge_feat = use_edge_feat
        
        self.hiddens = hiddens
        self.layers_ = nn.ModuleList([])
        layer0 = GraphConvolutionLayer(input_dim=input_dim,
                                  output_dim=hiddens[0], conv='transformer', dropout=dropout,
                                  activation=nn.ReLU(), add_self_loop=False, bn=True, is_first=True, is_last=False)
        self.layers_.append(layer0)
        nhiddens = len(hiddens)
        for _ in range(1,nhiddens):
            if _ != nhiddens-1: # last layer 
                layertemp = GraphConvolutionLayer(input_dim=hiddens[_-1],
                                        output_dim=hiddens[_], conv='transformer', dropout=dropout,
                                        activation=nn.ReLU(), add_self_loop=False, bn=True, is_first=False, is_last=False)
            else: 
                layertemp = GraphConvolutionLayer(input_dim=hiddens[_-1],
                                      output_dim=hiddens[_], conv='transformer', dropout=dropout,
                                      activation=nn.ReLU(), add_self_loop=False, bn=True, is_first=False, is_last=True)
            self.layers_.append(layertemp)

    def forward(self, data, training=None):
        x, edge_index = data.node_feat.to(torch.float), data.edge_index
        if self.use_edge_feat: 
            from utils.data.protein import edge_feature_gearnet_pyg, annotate_distmat_pyg
            data = annotate_distmat_pyg(data)
            data = edge_feature_gearnet_pyg(data)
        else: 
            data.edge_feat = None
        for layer in self.layers_:
            x = layer(x, edge_index, training, edge_attr=data.edge_feat)

        return x 

class GraphTransformerV1(nn.Module): # submodule 
    def __init__(self, input_dim, hiddens, output_dim, dropout, **kwargs):
        super(GraphTransformerV1, self).__init__(**kwargs)

        self.hiddens = hiddens
        self.layers_ = nn.ModuleList([])
        layer0 = GraphConvolutionLayer(input_dim=input_dim,
                                  output_dim=hiddens[0], conv='transformer-v1', dropout=dropout,
                                  activation=nn.ReLU(), add_self_loop=add_self_loop, bn=bn)
        self.layers_.append(layer0)
        nhiddens = len(hiddens)
        for _ in range(1,nhiddens):
            layertemp = GraphConvolutionLayer(input_dim=hiddens[_-1],
                                      output_dim=hiddens[_], conv='transformer-v1', dropout=dropout,
                                      activation=nn.ReLU(), add_self_loop=add_self_loop, bn=bn)
            self.layers_.append(layertemp)

    def forward(self, data, training=None):
        x, edge_index = data.node_feat.to(torch.float), data.edge_index
        for layer in self.layers_:
            x = layer(x, edge_index, training)
        return x 