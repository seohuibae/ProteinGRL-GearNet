# https://graphein.ai/notebooks/residue_graphs.html
# https://graphein.ai/notebooks/alphafold_protein_graph_tutorial.html
# http://rasbt.github.io/biopandas/tutorials/Working_with_PDB_Structures_in_DataFrames/ 
# http://rasbt.github.io/biopandas/api_subpackages/biopandas.pdb/

# protein_path = '../swissprot/pdb/AF-A1B2X1-F1-model_v3.pdb'
# protein_path = '../swissprot/pdb/AF-Q003G9-F1-model_v3.pdb'
protein_path = '../swissprot/pdb/AF-B1YH84-F1-model_v3.pdb.gz'

# from biopandas.pdb import PandasPdb
from data.biopandas.pdb import PandasPdb
import matplotlib.pyplot as plt

atomic_df = PandasPdb().read_pdb(protein_path) # df={'ATOM'} .get(records='ATOM')
# atomic_df.df.keys = {'ATOM'}

# print(atomic_df)
# print(atomic_df.df)
# input()
# # atomic_df.df['ATOM']['b_factor'].plot(kind='hist')
# # print(atomic_df.df['ATOM'].head())
# # print(atomic_df.df.keys())
# print(atomic_df.df['ATOM'])
# print(atomic_df.pdb_text.splitlines(True))
# print(atomic_df.get_model(1))
# exit()
# # print(atomic_df.label_models())
# idxs = atomic_df.get_model_start_end()
# print(idxs.start_idx.values, idxs.end_idx.values)
# print(idxs.model_idx)
# pdb_df = atomic_df.df['ATOM']
# print(pdb_df)
# print(pdb_df.loc[pdb_df['model_id']==model_idx])
# exit()
# print(pdb_df.line_idx.values)
# # m = atomic_df.get_model(1)
# # print(m)
# input()

print(atomic_df.get_model(1))
input()

from graphein.protein.config import ProteinGraphConfig
# from graphein.protein.graphs import construct_graph, construct_graphs_mp
from data.graphein.protein.graphs import construct_graph, construct_graphs_mp
from graphein.protein.edges.atomic import add_atomic_edges


# Load the default config
# c = ProteinGraphConfig(granularity='atom')
params_to_change = {"granularity": "atom", "edge_construction_functions": [add_atomic_edges], "node_metadata_functions": None}
c = ProteinGraphConfig(**params_to_change)
print(c)
input()
# Construct the graph!
g = construct_graph(pdb_path=protein_path, config=c) # config=c, 
# g = construct_graph(uniprot_id='A1B2X1', config=c, model_index=0)
# g = construct_graphs_mp(pdb_path_it=[protein_path], config=c, return_dict=True)
print(g)
input()

from graphein.protein.edges.distance import add_aromatic_interactions, add_cation_pi_interactions, add_hydrophobic_interactions, add_ionic_interactions

config = ProteinGraphConfig(edge_construction_functions=[add_aromatic_interactions,
                                                         add_cation_pi_interactions,
                                                         add_hydrophobic_interactions,
                                                         add_ionic_interactions])

g = construct_graph(pdb_path=protein_path, config=config)

plotly_protein_structure_graph(g, colour_edges_by="kind", colour_nodes_by="residue_name", label_node_ids=False, node_size_multiplier=2, node_size_min=5)