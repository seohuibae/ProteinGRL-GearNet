import torch.nn as nn 
import torch.nn.functional as F 
from models.layers import pooling_layer

def build_model(args, encoder):
    try:
        hiddens = [int(s) for s in args.hiddens.split('-')]
    except:
        hiddens =[] 

    output_dim_dict={
        'swissprot': hiddens[-1],
        'EC': 1, # 538 takss 
        'GO-BP': 1, 
        'GO-MF': 1, 
        'GO-CC': 1,
        'FC-Fold': 1195,
        'FC-Super': 1195, 
        'FC-Fam': 1195, 
        'RX': 384,
    }
    model_name = args.model 
    return Net(encoder, output_dim_dict[args.dataset])

def build_encoder(args): 
    model_name = args.model 
    input_dim = 21 # node feat dimension 
    try:
        hiddens = [int(s) for s in args.hiddens.split('-')]
    except:
        hiddens =[] 
    ########################################
    if model_name == 'gcn': # gcn, gvp, 3dcnn_mqa, graphqa, new ieconv
        from models.encoders import GCN
        encoder = GCN(input_dim, hiddens, hiddens[-1], args.dropout)
    elif model_name == 'transformer': # cnn, resnet, lstm, transformer
        raise NotImplementedError
    elif model_name == 'gearnet':
        from models.encoders import GearNet
        encoder = GearNet(input_dim, hiddens, hiddens[-1], args.dropout)
    elif model_name == 'gearnet-ieconv':
        from models.encoders import GearNetIEConv
        encoder = GearNetIEConv(input_dim, hiddens, hiddens[-1], args.dropout)
    elif model_name == 'gearnet-edge': 
        from models.encoders import GearNetEdge
        encoder = GearNetEdge(input_dim, hiddens, hiddens[-1], args.dropout)
    elif model_name == 'gearnet-edge-ieconv': 
        from models.encoders import GearNetEdgeIEConv
        encoder = GearNetEdgeIEConv(input_dim, hiddens, hiddens[-1], args.dropout)
    else: 
        raise NotImplementedError
    ######################################## 
    return encoder 

class Net(nn.Module): 
    def __init__(self, encoder, output_dim, **kwargs):
        super(Net, self).__init__(**kwargs)
        self.encoder = encoder 
        self.pooling = pooling_layer['mean']
        self.mlp_heads = nn.Sequential(nn.Linear(self.encoder.hiddens[-1], self.encoder.hiddens[-1]), nn.Dropout(0.25), nn.Linear(self.encoder.hiddens[-1], output_dim))

    def forward(self, data, training=None): 
        x = self.encoder(data)
        x = self.pooling(x, data.batch)
        x = self.mlp_heads(x)
        return F.log_softmax(x,dim=-1)
            
