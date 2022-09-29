"""
Appendix C.1 Protein Graph Construction
"""
import itertools
import logging
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import copy 

from graphein.protein.edges.atomic import add_atomic_edges
from graphein.protein.edges.distance import add_sequence_distance_edges, add_distance_threshold, add_k_nn_edges, add_distance_to_edges

def add_sequential_edges(G: nx.Graph, d_seq: int=3, name: str='sequence_edge'): 
    for d in range(-d_seq+1, d_seq, 1):
        add_sequence_distance_edges(G, d, name+f'_{d}')
    return G 

# spatial edge
def add_radius_edges(G: nx.Graph, d_radius: float=10.0, d_long: float=5.0): 
    add_distance_threshold(G, long_interaction_threshold=d_long, threshold=d_radius)
    return G 

# spatial edge
def add_k_nearest_neighbor_edges(G: nx.Graph, k: int=10, d_long: float=5.0): 
    add_k_nn_edges(G, long_interaction_threshold=d_long, k=k, mode='connectivity',  metric='minkowski', p=2, include_self=False) # euclidean distance\
    return G 

