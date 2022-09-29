"""
ProteinGraphConfig

"""
from graphein.protein.config import ProteinGraphConfig
from graphein.ml.conversion import GraphFormatConvertor

from utils.data.protein import add_sequential_edges, add_radius_edges, add_k_nearest_neighbor_edges
from utils.data.protein import amino_acid_one_hot_additional 
from utils.data.protein import node_features_edge_one_hot_sequential_spatial_distance

params_to_change = {"granularity": "centroids",
"edge_construction_functions": [add_sequential_edges, add_radius_edges, add_k_nearest_neighbor_edges],
"node_metadata_functions": [amino_acid_one_hot_additional]}

graphein_config = ProteinGraphConfig(**params_to_change) 
columns = [
            "b_factor",
            "chain_id",
            "coords", 
            "edge_index",
            "kind",
            "name",
            "node_id",
            "residue_name",
            "node_feat"
        ]
graph_format_convertor =  GraphFormatConvertor(src_format="nx", dst_format="pyg", columns=columns)

# "edge_metadata_functions": [node_features_edge_one_hot_sequential_spatial_distance]}

# "dist_mat",
# "node_feat", 
# "edge_feat"


# {'granularity': 'CA',
# 'keep_hets': False,
# 'insertions': False,
# 'pdb_dir': PosixPath('../examples/pdbs'),
# 'verbose': False,
# 'exclude_waters': True,
# 'deprotonate': False,
# 'protein_df_processing_functions': None,
# 'edge_construction_functions': [<function graphein.protein.edges.distance.add_peptide_bonds(G: 'nx.Graph') -> 'nx.Graph'>],
# 'node_metadata_functions': [<function graphein.protein.features.nodes.amino_acid.meiler_embedding(n, d, return_array: bool = False) -> Union[pandas.core.series.Series, <built-in function array>]>],
# 'edge_metadata_functions': None,
# 'graph_metadata_functions': None,
# 'get_contacts_config': None,
# 'dssp_config': None}
