from typing import Optional, Tuple, Union

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch.nn import Parameter as Param
from torch_scatter import scatter
from torch_sparse import SparseTensor, masked_select_nnz, matmul

from torch_geometric.nn.conv import MessagePassing, GCNConv
from torch_geometric.nn.conv.rgcn_conv import RGCNConv, FastRGCNConv
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn.dense.linear import Linear

class SimplfiedIEConv(GCNConv): # Modified in (Zhang et al., 2022), Originally (Hermosilla et al., 2021) 
    def __init__(self, in_channels: int, out_channels: int, edge_in_channels: int, edge_out_channels: int, 
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs): 
        kwargs.setdefault('aggr', 'add')
        super().__init__(in_channels, out_channels,improved, cached,
                 add_self_loops, normalize, bias, **kwargs) 
        self.lin_edge = Linear(edge_in_channels, edge_out_channels, bias=False, weight_initializer='glorot')
        self.reset_parameters_edge()

    def reset_parameters_edge(self):
        self.lin_edge.reset_parameters()
        
    def forward(self, x, edge_index, edge_weight) -> Tensor:
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = self.lin(x)
        edge_weight=self.lin_edge(edge_weight) # TODO 
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out = out + self.bias

        return out


class GearNetConv(RGCNConv):
    from utils.data.protein import EDGE_TYPE_GEARNET
    def __init__(self, in_channels: int, out_channels: int, num_relations: int = len(EDGE_TYPE_GEARNET),
                aggr: str = 'add', root_weight: bool = False, bias: bool = True, **kwargs):
        super(GearNetConv, self).__init__(in_channels, out_channels, num_relations, aggr=aggr, bias=bias, root_weight=root_weight, **kwargs)

class GearNetEdgeConv(FastRGCNConv): # (Zhang et al., 2022)
    from utils.data.protein import EDGE_TYPE_GEARNET
    def __init__(self, in_channels: int, out_channels: int, num_relations: int = len(EDGE_TYPE_GEARNET),
                aggr: str = 'add', root_weight: bool = False, bias: bool = True, **kwargs):
        super(GearNetEdgeConv, self).__init__(in_channels, out_channels, num_relations, aggr=aggr, bias=bias, root_weight=root_weight,  **kwargs)
        self.fc_message = nn.Linear(in_channels, out_channels) # TODO dim

    def message(self, x_j: Tensor, edge_type_ptr: OptTensor, message_jir: OptTensor) -> Tensor:
        if edge_type_ptr is not None:
            return segment_matmul(x_j, edge_type_ptr, self.weight) + self.fc_message(message_jir)
        return x_j + self.fc_message(message_jir) # sum


# ###############
### for pyg built-in convs
# from typing import Optional, Tuple, Union

# import torch
# import torch.nn.functional as F
# from torch import Tensor
# from torch.nn import Parameter
# from torch.nn import Parameter as Param
# from torch_scatter import scatter
# from torch_sparse import SparseTensor, masked_select_nnz, matmul

# from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.typing import Adj, OptTensor

# from .inits import glorot, zeros


# try:
#     from pyg_lib.ops import segment_matmul  # noqa
#     _WITH_PYG_LIB = True
# except ImportError:
#     _WITH_PYG_LIB = False

#     def segment_matmul(inputs: Tensor, ptr: Tensor, other: Tensor) -> Tensor:
#         raise NotImplementedError

# def masked_edge_index(edge_index, edge_mask):
#     if isinstance(edge_index, Tensor):
#         return edge_index[:, edge_mask]
#     else:
#         return masked_select_nnz(edge_index, edge_mask, layout='coo')
# ####
# class RGCNConv(MessagePassing):
#     def __init__(
#         self,
#         in_channels: Union[int, Tuple[int, int]],
#         out_channels: int,
#         num_relations: int,
#         num_bases: Optional[int] = None,
#         num_blocks: Optional[int] = None,
#         aggr: str = 'mean',
#         root_weight: bool = True,
#         is_sorted: bool = False,
#         bias: bool = True,
#         **kwargs,
#     ):
#         kwargs.setdefault('aggr', aggr)
#         super().__init__(node_dim=0, **kwargs)
#         self._WITH_PYG_LIB = torch.cuda.is_available() and _WITH_PYG_LIB

#         if num_bases is not None and num_blocks is not None:
#             raise ValueError('Can not apply both basis-decomposition and '
#                              'block-diagonal-decomposition at the same time.')

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.num_relations = num_relations
#         self.num_bases = num_bases
#         self.num_blocks = num_blocks
#         self.is_sorted = is_sorted

#         if isinstance(in_channels, int):
#             in_channels = (in_channels, in_channels)
#         self.in_channels_l = in_channels[0]

#         if num_bases is not None:
#             self.weight = Parameter(
#                 torch.Tensor(num_bases, in_channels[0], out_channels))
#             self.comp = Parameter(torch.Tensor(num_relations, num_bases))

#         elif num_blocks is not None:
#             assert (in_channels[0] % num_blocks == 0
#                     and out_channels % num_blocks == 0)
#             self.weight = Parameter(
#                 torch.Tensor(num_relations, num_blocks,
#                              in_channels[0] // num_blocks,
#                              out_channels // num_blocks))
#             self.register_parameter('comp', None)

#         else:
#             self.weight = Parameter(
#                 torch.Tensor(num_relations, in_channels[0], out_channels))
#             self.register_parameter('comp', None)

#         if root_weight:
#             self.root = Param(torch.Tensor(in_channels[1], out_channels))
#         else:
#             self.register_parameter('root', None)

#         if bias:
#             self.bias = Param(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)

#         self.reset_parameters()

#     def reset_parameters(self):
#         glorot(self.weight)
#         glorot(self.comp)
#         glorot(self.root)
#         zeros(self.bias)

#     def forward(self, x: Union[OptTensor, Tuple[OptTensor, Tensor]],
#                 edge_index: Adj, edge_type: OptTensor = None):
#         # Convert input features to a pair of node features or node indices.
#         x_l: OptTensor = None
#         if isinstance(x, tuple):
#             x_l = x[0]
#         else:
#             x_l = x
#         if x_l is None:
#             x_l = torch.arange(self.in_channels_l, device=self.weight.device)

#         x_r: Tensor = x_l
#         if isinstance(x, tuple):
#             x_r = x[1]

#         size = (x_l.size(0), x_r.size(0))

#         if isinstance(edge_index, SparseTensor):
#             edge_type = edge_index.storage.value()
#         assert edge_type is not None

#         # propagate_type: (x: Tensor, edge_type_ptr: OptTensor)
#         out = torch.zeros(x_r.size(0), self.out_channels, device=x_r.device)

#         weight = self.weight
#         if self.num_bases is not None:  # Basis-decomposition =================
#             weight = (self.comp @ weight.view(self.num_bases, -1)).view(
#                 self.num_relations, self.in_channels_l, self.out_channels)

#         if self.num_blocks is not None:  # Block-diagonal-decomposition =====

#             if x_l.dtype == torch.long and self.num_blocks is not None:
#                 raise ValueError('Block-diagonal decomposition not supported '
#                                  'for non-continuous input features.')

#             for i in range(self.num_relations):
#                 tmp = masked_edge_index(edge_index, edge_type == i)
#                 h = self.propagate(tmp, x=x_l, edge_type_ptr=None, size=size)
#                 h = h.view(-1, weight.size(1), weight.size(2))
#                 h = torch.einsum('abc,bcd->abd', h, weight[i])
#                 out = out + h.contiguous().view(-1, self.out_channels)

#         else:  # No regularization/Basis-decomposition ========================
#             if self._WITH_PYG_LIB and isinstance(edge_index, Tensor):
#                 if not self.is_sorted:
#                     if (edge_type[1:] < edge_type[:-1]).any():
#                         edge_type, perm = edge_type.sort()
#                         edge_index = edge_index[:, perm]
#                 edge_type_ptr = torch.ops.torch_sparse.ind2ptr(
#                     edge_type, self.num_relations)
#                 out = self.propagate(edge_index, x=x_l,
#                                      edge_type_ptr=edge_type_ptr, size=size)
#             else:
#                 for i in range(self.num_relations):
#                     tmp = masked_edge_index(edge_index, edge_type == i)
#                     h = self.propagate(tmp, x=x_l, edge_type_ptr=None,
#                                         size=size)
#                     out = out + (h @ weight[i])
#                     print(h.shape)
#                     print((h @ weight[i]).shape)
#                     print(out.shape)

#         root = self.root
#         if root is not None:
#             out = out + (root[x_r] if x_r.dtype == torch.long else x_r @ root)

#         if self.bias is not None:
#             out = out + self.bias

#         return out


#     def message(self, x_j: Tensor, edge_type_ptr: OptTensor) -> Tensor:
#         if edge_type_ptr is not None:
#             return segment_matmul(x_j, edge_type_ptr, self.weight)

#         return x_j

#     def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
#         adj_t = adj_t.set_value(None)
#         return matmul(adj_t, x, reduce=self.aggr)

#     def __repr__(self) -> str:
#         return (f'{self.__class__.__name__}({self.in_channels}, '
#                 f'{self.out_channels}, num_relations={self.num_relations})')


# class FastRGCNConv(RGCNConv):
#     r"""See :class:`RGCNConv`."""
#     def forward(self, x: Union[OptTensor, Tuple[OptTensor, Tensor]],
#                 edge_index: Adj, edge_type: OptTensor = None):
#         """"""
#         self.fuse = False
#         assert self.aggr in ['add', 'sum', 'mean']

#         # Convert input features to a pair of node features or node indices.
#         x_l: OptTensor = None
#         if isinstance(x, tuple):
#             x_l = x[0]
#         else:
#             x_l = x
#         if x_l is None:
#             x_l = torch.arange(self.in_channels_l, device=self.weight.device)

#         x_r: Tensor = x_l
#         if isinstance(x, tuple):
#             x_r = x[1]

#         size = (x_l.size(0), x_r.size(0))

#         # propagate_type: (x: Tensor, edge_type: OptTensor)
#         out = self.propagate(edge_index, x=x_l, edge_type=edge_type, size=size)

#         root = self.root
#         if root is not None:
#             out = out + (root[x_r] if x_r.dtype == torch.long else x_r @ root)

#         if self.bias is not None:
#             out = out + self.bias

#         return out


#     def message(self, x_j: Tensor, edge_type: Tensor,
#                 edge_index_j: Tensor) -> Tensor:
#         weight = self.weight
#         if self.num_bases is not None:  # Basis-decomposition =================
#             weight = (self.comp @ weight.view(self.num_bases, -1)).view(
#                 self.num_relations, self.in_channels_l, self.out_channels)

#         if self.num_blocks is not None:  # Block-diagonal-decomposition =======
#             if x_j.dtype == torch.long:
#                 raise ValueError('Block-diagonal decomposition not supported '
#                                  'for non-continuous input features.')

#             weight = weight[edge_type].view(-1, weight.size(2), weight.size(3))
#             x_j = x_j.view(-1, 1, weight.size(1))
#             return torch.bmm(x_j, weight).view(-1, self.out_channels)

#         else:  # No regularization/Basis-decomposition ========================
#             if x_j.dtype == torch.long:
#                 weight_index = edge_type * weight.size(1) + edge_index_j
#                 return weight.view(-1, self.out_channels)[weight_index]

#             return torch.bmm(x_j.unsqueeze(-2), weight[edge_type]).squeeze(-2)

#     def aggregate(self, inputs: Tensor, edge_type: Tensor, index: Tensor,
#                   dim_size: Optional[int] = None) -> Tensor:

#         # Compute normalization in separation for each `edge_type`.
#         if self.aggr == 'mean':
#             norm = F.one_hot(edge_type, self.num_relations).to(torch.float)
#             norm = scatter(norm, index, dim=0, dim_size=dim_size)[index]
#             norm = torch.gather(norm, 1, edge_type.view(-1, 1))
#             norm = 1. / norm.clamp_(1.)
#             inputs = norm * inputs

#         return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size)