"""
Appendix C.1 Protein Graph Construction
"""
import itertools
import logging
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Union, Any

import networkx as nx
import numpy as np
import pandas as pd

from .features_nodes_gearnet import amino_acid_one_hot_additional
from graphein.utils.utils import onek_encoding_unk

# "sequence_edge_-2",
    # "sequence_edge_-1",
EDGE_TYPE_GEARNET = [
    "sequence_edge_0",
    "sequence_edge_1",
    "sequence_edge_2",
    "distance_threshold",
    "k_nn"
]
EDGE_FEAT_DIM = 49 

def node_features_edge_one_hot_sequential_spatial_distance(
    u, v,
    d: Dict[str, Any],
    return_array: bool = True,
    allowable_set: Optional[List[str]] = EDGE_TYPE_GEARNET,
) -> Union[pd.Series, np.ndarray]:
    """Adds a one-hot encoding of amino acid types as a node attribute.
    :param u: node name j, this is unused and only included for compatibility with the other functions
    :type u: str
    :param v: node name i, this is unused and only included for compatibility with the other functions
    :type v: str
    :param d: edge data
    :type d: Dict[str, Any]
    :param return_array: If True, returns a numpy array of one-hot encoding, otherwise returns a pd.Series. Default is True.
    :type return_array: bool
    :param allowable_set: Specifies vocabulary of amino acids. Default is None (which uses `graphein.protein.resi_atoms.STANDARD_AMINO_ACIDS`).
    :return: One-hot encoding of amino acid types
    :rtype: Union[pd.Series, np.ndarray]
    """
    """
    The feature f(i,j,r) for an edge (i, j, r) is the concatenation of
    the node features of two end nodes, the one-hot encoding of
    the edge type, and the sequential and spatial distances between
    them:
    """ 
    def parse_node_name(str):
        return {'residue_name': str.split(':')[1]}
    # def parse_kind_to_seq_distance(u, v): 
    def parse_residue_name_to_seq_distance(u,v):
        return int(v.split(':')[-1])-int(u.split(':')[-1])
        
    node_feature_u = amino_acid_one_hot_additional(u, parse_node_name(u))
    node_feature_v = amino_acid_one_hot_additional(v, parse_node_name(v))
    spatial_dist = d["distance"]
    sequence_dist = parse_residue_name_to_seq_distance(u,v)
    # sequence_dist = parse_kind_to_seq_distance(u,v)
    # edge_one_hot = onek_encoding_unk(sequence_dist, allowable_set)
    edge_one_hot = onek_encoding_unk(d["kind"], allowable_set)
    
    node_feature_u = np.array(node_feature_u)
    node_feature_v = np.array(node_feature_v)
    edge_one_hot = np.array(edge_one_hot)
    sequence_dist = np.array([sequence_dist])
    spatial_dist = np.array([spatial_dist])

    features = np.expand_dims(np.concatenate([node_feature_u, node_feature_v, edge_one_hot, sequence_dist, spatial_dist], axis=0),0)
    if return_array:
        features = np.array(features)
    else:
        features = pd.Series(features)
        features.index = allowable_set

    d["edge_feat"] = features
    
    return features
