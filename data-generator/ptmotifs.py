# In[Import]
import os
import numpy as np
import os.path as osp
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from tqdm import tqdm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import featgen
import synthetic_structsim

# In[Hyper-parameter]
# global_b = '0.333' # Set bias degree here
data_dir = f'./data/PT-Motifs/raw/'
os.makedirs(data_dir, exist_ok=True)
figsize = (8, 6)

# In[Perturb]
def perturb(graph_list, p, id=None):
    """ Perturb the list of (sparse) graphs by adding/removing edges.
    Args:
        p: proportion of added edges based on current number of edges.
    Returns:
        A list of graphs that are perturbed from the original graphs.
    """
    perturbed_graph_list = []
    for G_original in graph_list:
        G = G_original.copy()
        edge_count = int(G.number_of_edges() * p)
        # randomly add the edges between a pair of nodes without an edge.
        for _ in range(edge_count):
            while True:
                u = np.random.randint(0, G.number_of_nodes())
                v = np.random.randint(0, G.number_of_nodes())
                if (not G.has_edge(u, v)) and (u != v):
                    break
            if (not id == None) and (id[u]==0 or id[v]==0): 
                G.add_edge(u, v)
        perturbed_graph_list.append(G)
    return perturbed_graph_list

# In[Find Grid]
def find_gd(edge_index, ids):
    row, col = edge_index
    gd = np.array(ids[row] > 0, dtype=np.float64) *\
        np.array(ids[col] > 0, dtype=np.float64)
    
    return gd

# In[Motif: Star]
def get_star(basis_type, nb_shapes=80, width_basis=8, 
              feature_generator=None, m=3, draw=False):
    """ Synthetic Graph:
    Start with a tree and attach DIAMOND-shaped subgraphs.
    """
    list_shapes = [["diamond"]] * nb_shapes # house

    if draw:
        plt.figure(figsize=figsize)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, rdm_basis_plugins=True
    )
    G = perturb([G], 0.00, id=role_id)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    return G, role_id, name


# In[Motif: Diamond]
def get_diamond(basis_type, nb_shapes=80, width_basis=8, 
              feature_generator=None, m=3, draw=False):
    """ Synthetic Graph:
    Start with a tree and attach DIAMOND-shaped subgraphs.
    """
    list_shapes = [["diamond"]] * nb_shapes # house

    if draw:
        plt.figure(figsize=figsize)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, rdm_basis_plugins=True
    )
    G = perturb([G], 0.00, id=role_id)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    return G, role_id, name

# In[Motif: House]
def get_house(basis_type, nb_shapes=80, width_basis=8, 
              feature_generator=None, m=3, draw=True):
    """ Synthetic Graph:
    Start with a tree and attach HOUSE-shaped subgraphs.
    """
    list_shapes = [["house"]] * nb_shapes # house

    if draw:
        plt.figure(figsize=figsize)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, rdm_basis_plugins=True
    )
    G = perturb([G], 0.00, id=role_id)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    return G, role_id, name

# In[Motif: Cycle]
def get_cycle(basis_type, nb_shapes=80, width_basis=8, 
              feature_generator=None, m=3, draw=True):
    """ Synthetic Graph:
    Start with a tree and attach cycle-shaped (directed edges) subgraphs.
    """
    list_shapes = [["dircycle"]] * nb_shapes

    if draw:
        plt.figure(figsize=figsize)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, rdm_basis_plugins=True
    )
    G = perturb([G], 0.00, id=role_id)[0]       # 0.05 original

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    return G, role_id, name

# In[Motif: Crane]
def get_crane(basis_type, nb_shapes=80, width_basis=8, feature_generator=None, m=3, draw=True):
    """ Synthetic Graph:
    Start with a tree and attach crane-shaped subgraphs.
    """
    list_shapes = [["varcycle"]] * nb_shapes   # crane

    if draw:
        plt.figure(figsize=figsize)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, rdm_basis_plugins=True
    )
    G = perturb([G], 0.00, id=role_id)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    return G, role_id, name

# In[Base]
def graph_stats(base_num):
    if base_num == 1:
        base = 'tree'
        width_basis=np.random.choice(range(3))
    if base_num == 2:
        base = 'ladder'
        width_basis=np.random.choice(range(8,12))
    if base_num == 3:
        base = 'wheel'
        width_basis=np.random.choice(range(15,20))
    if base_num == 4:
        base = 'clique'
        width_basis=np.random.choice(range(15,20))
    if base_num == 5:
        base = 'ba'
        width_basis=np.random.choice(range(20,25))
    return base, width_basis

def graph_stats_large(base_num):
    if base_num == 1:
        base = 'tree'
        width_basis=np.random.choice(range(3,6))
    if base_num == 2:
        base = 'ladder'
        width_basis=np.random.choice(range(30,50))
    if base_num == 3:
        base = 'wheel'
        width_basis=np.random.choice(range(60,80))
    if base_num == 4:
        base = 'clique'
        width_basis=np.random.choice(range(60,80))
    if base_num == 5:
        base = 'ba'
        width_basis=np.random.choice(range(60,80))
    return base, width_basis

# -------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------- #
# In[Training set]
edge_index_list, label_list = [], []
ground_truth_list, role_id_list, pos_list = [], [], []
# bias = float(global_b)

# C = 5, normal.
e_mean, n_mean = [], []
for _ in tqdm(range(5000)):
    # base_num = np.random.choice([1,2,3], p=[(1-bias)/2,(1-bias)/2,bias])
    base_num = np.random.choice([1,2,3,4,5])
    base, width_basis = graph_stats(base_num)
    
    G, role_id, name = get_star(basis_type=base, nb_shapes=1, 
        width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(2)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("#Graphs: %d    #Nodes: %.2f    #Edges: %.2f " % (len(ground_truth_list), 
    np.mean(n_mean), np.mean(e_mean)))

# C = 5, large.
e_mean, n_mean = [], []
for _ in tqdm(range(5000)):
    # base_num = np.random.choice([1,2,3], p=[(1-bias)/2,(1-bias)/2,bias])
    base_num = np.random.choice([1,2,3,4,5])
    base, width_basis = graph_stats_large(base_num)
    
    G, role_id, name = get_star(basis_type=base, nb_shapes=1, 
        width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(2)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("#Graphs: %d    #Nodes: %.2f    #Edges: %.2f " % (len(ground_truth_list), 
    np.mean(n_mean), np.mean(e_mean)))

# C = 4, normal.
e_mean, n_mean = [], []
for _ in tqdm(range(5000)):
    # base_num = np.random.choice([1,2,3], p=[(1-bias)/2,(1-bias)/2,bias])
    base_num = np.random.choice([1,2,3,4,5])
    base, width_basis = graph_stats(base_num)
    
    G, role_id, name = get_diamond(basis_type=base, nb_shapes=1, 
        width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(2)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("#Graphs: %d    #Nodes: %.2f    #Edges: %.2f " % (len(ground_truth_list), 
    np.mean(n_mean), np.mean(e_mean)))

# C = 4, large.
e_mean, n_mean = [], []
for _ in tqdm(range(5000)):
    # base_num = np.random.choice([1,2,3], p=[(1-bias)/2,(1-bias)/2,bias])
    base_num = np.random.choice([1,2,3,4,5])
    base, width_basis = graph_stats_large(base_num)
    
    G, role_id, name = get_diamond(basis_type=base, nb_shapes=1, 
        width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(2)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("#Graphs: %d    #Nodes: %.2f    #Edges: %.2f " % (len(ground_truth_list), 
    np.mean(n_mean), np.mean(e_mean)))

# C = 1, normal.
e_mean, n_mean = [], []
for _ in tqdm(range(5000)):
    # base_num = np.random.choice([1,2,3], p=[bias,(1-bias)/2,(1-bias)/2])
    base_num = np.random.choice([1,2,3,4,5])
    base, width_basis = graph_stats(base_num)

    G, role_id, name = get_cycle(basis_type=base, nb_shapes=1,
        width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(0)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("#Graphs: %d    #Nodes: %.2f    #Edges: %.2f " % (len(ground_truth_list), 
    np.mean(n_mean), np.mean(e_mean)))

# C = 1, large.
e_mean, n_mean = [], []
for _ in tqdm(range(5000)):
    # base_num = np.random.choice([1,2,3], p=[bias,(1-bias)/2,(1-bias)/2])
    base_num = np.random.choice([1,2,3,4,5])
    base, width_basis = graph_stats_large(base_num)

    G, role_id, name = get_cycle(basis_type=base, nb_shapes=1,
        width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(0)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("#Graphs: %d    #Nodes: %.2f    #Edges: %.2f " % (len(ground_truth_list), 
    np.mean(n_mean), np.mean(e_mean)))

# C = 2, normal.
e_mean, n_mean = [], []
for _ in tqdm(range(5000)):
    # base_num = np.random.choice([1,2,3], p=[(1-bias)/2,bias,(1-bias)/2])
    base_num = np.random.choice([1,2,3,4,5])
    base, width_basis = graph_stats(base_num)

    G, role_id, name = get_house(basis_type=base, nb_shapes=1, 
        width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(1)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("#Graphs: %d    #Nodes: %.2f    #Edges: %.2f " % (len(ground_truth_list), 
    np.mean(n_mean), np.mean(e_mean)))

# C = 2, large.
e_mean, n_mean = [], []
for _ in tqdm(range(5000)):
    # base_num = np.random.choice([1,2,3], p=[(1-bias)/2,bias,(1-bias)/2])
    base_num = np.random.choice([1,2,3,4,5])
    base, width_basis = graph_stats_large(base_num)

    G, role_id, name = get_house(basis_type=base, nb_shapes=1, 
        width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(1)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("#Graphs: %d    #Nodes: %.2f    #Edges: %.2f " % (len(ground_truth_list), 
    np.mean(n_mean), np.mean(e_mean)))

# C = 3, normal.
e_mean, n_mean = [], []
for _ in tqdm(range(5000)):
    # base_num = np.random.choice([1,2,3], p=[(1-bias)/2,(1-bias)/2,bias])
    base_num = np.random.choice([1,2,3,4,5])
    base, width_basis = graph_stats(base_num)
    
    G, role_id, name = get_crane(basis_type=base, nb_shapes=1, 
        width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(2)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("#Graphs: %d    #Nodes: %.2f    #Edges: %.2f " % (len(ground_truth_list), 
    np.mean(n_mean), np.mean(e_mean)))

# C = 3, large.
e_mean, n_mean = [], []
for _ in tqdm(range(5000)):
    # base_num = np.random.choice([1,2,3], p=[(1-bias)/2,(1-bias)/2,bias])
    base_num = np.random.choice([1,2,3,4,5])
    base, width_basis = graph_stats_large(base_num)
    
    G, role_id, name = get_crane(basis_type=base, nb_shapes=1, 
        width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(2)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("#Graphs: %d    #Nodes: %.2f    #Edges: %.2f " % (len(ground_truth_list), 
    np.mean(n_mean), np.mean(e_mean)))

# In[Save]
np.save(osp.join(data_dir, 'train.npy'), 
    (edge_index_list, label_list, ground_truth_list, role_id_list, pos_list))
# -------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------- #


# -------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------- #
# In[Validation set]
edge_index_list, label_list = [], []
ground_truth_list, role_id_list, pos_list = [], [], []
# bias = float(global_b)

# C = 1, normal.
e_mean, n_mean = [], []
for _ in tqdm(range(1000)):
    base_num = np.random.choice([1,2,3,4,5])
    base, width_basis = graph_stats(base_num)

    G, role_id, name = get_cycle(basis_type=base, nb_shapes=1, 
        width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(0)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("#Graphs: %d    #Nodes: %.2f    #Edges: %.2f " % (len(ground_truth_list), 
    np.mean(n_mean), np.mean(e_mean)))

# C = 1, large.
e_mean, n_mean = [], []
for _ in tqdm(range(1000)):
    base_num = np.random.choice([1,2,3,4,5])
    base, width_basis = graph_stats_large(base_num)

    G, role_id, name = get_cycle(basis_type=base, nb_shapes=1, 
        width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(0)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("#Graphs: %d    #Nodes: %.2f    #Edges: %.2f " % (len(ground_truth_list), 
    np.mean(n_mean), np.mean(e_mean)))

# C = 2, normal.
e_mean, n_mean = [], []
for _ in tqdm(range(1000)):
    base_num = np.random.choice([1,2,3,4,5])
    base, width_basis = graph_stats(base_num)

    G, role_id, name = get_house(basis_type=base, nb_shapes=1,
        width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(1)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("#Graphs: %d    #Nodes: %.2f    #Edges: %.2f " % (len(ground_truth_list), 
    np.mean(n_mean), np.mean(e_mean)))

# C = 2, large.
e_mean, n_mean = [], []
for _ in tqdm(range(1000)):
    base_num = np.random.choice([1,2,3,4,5])
    base, width_basis = graph_stats_large(base_num)

    G, role_id, name = get_house(basis_type=base, nb_shapes=1,
        width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(1)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("#Graphs: %d    #Nodes: %.2f    #Edges: %.2f " % (len(ground_truth_list), 
    np.mean(n_mean), np.mean(e_mean)))

# C = 3, normal.
e_mean, n_mean = [], []
for _ in tqdm(range(1000)):
    base_num = np.random.choice([1,2,3,4,5])
    base, width_basis = graph_stats(base_num)
    
    G, role_id, name = get_crane(basis_type=base, nb_shapes=1, 
        width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(2)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("# Graphs: %d    # Nodes: %.2f    # Edges: %.2f " % (len(ground_truth_list), 
    np.mean(n_mean), np.mean(e_mean)))

# C = 3, large.
e_mean, n_mean = [], []
for _ in tqdm(range(1000)):
    base_num = np.random.choice([1,2,3,4,5])
    base, width_basis = graph_stats_large(base_num)
    
    G, role_id, name = get_crane(basis_type=base, nb_shapes=1, 
        width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(2)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("# Graphs: %d    # Nodes: %.2f    # Edges: %.2f " % (len(ground_truth_list), 
    np.mean(n_mean), np.mean(e_mean)))

# C = 4, normal.
e_mean, n_mean = [], []
for _ in tqdm(range(1000)):
    base_num = np.random.choice([1,2,3,4,5])
    base, width_basis = graph_stats(base_num)
    
    G, role_id, name = get_diamond(basis_type=base, nb_shapes=1, 
        width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(2)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("# Graphs: %d    # Nodes: %.2f    # Edges: %.2f " % (len(ground_truth_list), 
    np.mean(n_mean), np.mean(e_mean)))

# C = 4, large.
e_mean, n_mean = [], []
for _ in tqdm(range(1000)):
    base_num = np.random.choice([1,2,3,4,5])
    base, width_basis = graph_stats_large(base_num)
    
    G, role_id, name = get_diamond(basis_type=base, nb_shapes=1, 
        width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(2)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("# Graphs: %d    # Nodes: %.2f    # Edges: %.2f " % (len(ground_truth_list), 
    np.mean(n_mean), np.mean(e_mean)))

# C = 5, normal.
e_mean, n_mean = [], []
for _ in tqdm(range(1000)):
    base_num = np.random.choice([1,2,3,4,5])
    base, width_basis = graph_stats(base_num)
    
    G, role_id, name = get_star(basis_type=base, nb_shapes=1, 
        width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(2)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("#Graphs: %d    #Nodes: %.2f    #Edges: %.2f " % (len(ground_truth_list), 
    np.mean(n_mean), np.mean(e_mean)))

# C = 5, large.
e_mean, n_mean = [], []
for _ in tqdm(range(1000)):
    base_num = np.random.choice([1,2,3,4,5])
    base, width_basis = graph_stats_large(base_num)
    
    G, role_id, name = get_star(basis_type=base, nb_shapes=1, 
        width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(2)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("#Graphs: %d    #Nodes: %.2f    #Edges: %.2f " % (len(ground_truth_list), 
    np.mean(n_mean), np.mean(e_mean)))

# In[Save]
np.save(osp.join(data_dir, 'val.npy'), 
    (edge_index_list, label_list, ground_truth_list, role_id_list, pos_list))
# -------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------- #


# -------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------- #
# In[Testing set]
edge_index_list, label_list = [], []
ground_truth_list, role_id_list, pos_list = [], [], []

# C = 1, normal.
e_mean, n_mean = [], []
for _ in tqdm(range(2000)):
    base_num = np.random.choice([1,2,3,4,5]) # uniform
    base, width_basis = graph_stats(base_num)

    G, role_id, name = get_cycle(basis_type=base, nb_shapes=1, 
        width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(0)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("#Graphs: %d    #Nodes: %.2f    #Edges: %.2f " % (len(ground_truth_list), 
    np.mean(n_mean), np.mean(e_mean)))

# C = 1, large.
e_mean, n_mean = [], []
for _ in tqdm(range(2000)):
    base_num = np.random.choice([1,2,3,4,5])
    base, width_basis = graph_stats_large(base_num)

    G, role_id, name = get_cycle(basis_type=base, nb_shapes=1, 
        width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(0)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("#Graphs: %d    #Nodes: %.2f    #Edges: %.2f " % (len(ground_truth_list), 
    np.mean(n_mean), np.mean(e_mean)))

# C = 2, normal.
e_mean, n_mean = [], []
for _ in tqdm(range(2000)):
    base_num = np.random.choice([1,2,3,4,5])
    base, width_basis = graph_stats(base_num)

    G, role_id, name = get_house(basis_type=base, nb_shapes=1, 
        width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(1)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("#Graphs: %d    #Nodes: %.2f    #Edges: %.2f " % (len(ground_truth_list), 
    np.mean(n_mean), np.mean(e_mean)))

# C = 2, large.
e_mean, n_mean = [], []
for _ in tqdm(range(2000)):
    base_num = np.random.choice([1,2,3,4,5])
    base, width_basis = graph_stats_large(base_num)

    G, role_id, name = get_house(basis_type=base, nb_shapes=1, 
        width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(1)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("#Graphs: %d    #Nodes: %.2f    #Edges: %.2f " % (len(ground_truth_list), 
    np.mean(n_mean), np.mean(e_mean)))

# C = 3, normal.
e_mean, n_mean = [], []
for _ in tqdm(range(2000)):
    base_num = np.random.choice([1,2,3,4,5])
    base, width_basis = graph_stats(base_num)

    G, role_id, name = get_crane(basis_type=base, nb_shapes=1, 
        width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(2)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("#Graphs: %d    #Nodes: %.2f    #Edges: %.2f " % (len(ground_truth_list), 
    np.mean(n_mean), np.mean(e_mean)))

# C = 3, large.
e_mean, n_mean = [], []
for _ in tqdm(range(2000)):
    base_num = np.random.choice([1,2,3,4,5])
    base, width_basis = graph_stats_large(base_num)

    G, role_id, name = get_crane(basis_type=base, nb_shapes=1, 
        width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(2)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("#Graphs: %d    #Nodes: %.2f    #Edges: %.2f " % (len(ground_truth_list), 
    np.mean(n_mean), np.mean(e_mean)))

# C = 4, normal.
e_mean, n_mean = [], []
for _ in tqdm(range(2000)):
    base_num = np.random.choice([1,2,3,4,5])
    base, width_basis = graph_stats(base_num)
    
    G, role_id, name = get_diamond(basis_type=base, nb_shapes=1, 
        width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(2)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("# Graphs: %d    # Nodes: %.2f    # Edges: %.2f " % (len(ground_truth_list), 
    np.mean(n_mean), np.mean(e_mean)))

# C = 4, large.
e_mean, n_mean = [], []
for _ in tqdm(range(2000)):
    base_num = np.random.choice([1,2,3,4,5])
    base, width_basis = graph_stats_large(base_num)
    
    G, role_id, name = get_diamond(basis_type=base, nb_shapes=1, 
        width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(2)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("# Graphs: %d    # Nodes: %.2f    # Edges: %.2f " % (len(ground_truth_list), 
    np.mean(n_mean), np.mean(e_mean)))

# C = 5, normal.
e_mean, n_mean = [], []
for _ in tqdm(range(2000)):
    base_num = np.random.choice([1,2,3,4,5])
    base, width_basis = graph_stats(base_num)
    
    G, role_id, name = get_star(basis_type=base, nb_shapes=1, 
        width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(2)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("#Graphs: %d    #Nodes: %.2f    #Edges: %.2f " % (len(ground_truth_list), 
    np.mean(n_mean), np.mean(e_mean)))

# C = 5, large.
e_mean, n_mean = [], []
for _ in tqdm(range(2000)):
    base_num = np.random.choice([1,2,3,4,5])
    base, width_basis = graph_stats_large(base_num)
    
    G, role_id, name = get_star(basis_type=base, nb_shapes=1, 
        width_basis=width_basis, feature_generator=None, m=3, draw=False)
    label_list.append(2)
    e_mean.append(len(G.edges))
    n_mean.append(len(G.nodes))

    role_id = np.array(role_id)
    edge_index = np.array(G.edges, dtype=np.int32).T

    role_id_list.append(role_id)
    edge_index_list.append(edge_index)
    pos_list.append(np.array(list(nx.spring_layout(G).values())))
    ground_truth_list.append(find_gd(edge_index, role_id))

print("#Graphs: %d    #Nodes: %.2f    #Edges: %.2f " % (len(ground_truth_list), 
    np.mean(n_mean), np.mean(e_mean)))

# In[Save]
np.save(osp.join(data_dir, 'test.npy'), 
    (edge_index_list, label_list, ground_truth_list, role_id_list, pos_list))