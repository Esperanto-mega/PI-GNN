# In[Import]
import pickle as pkl
import numpy as np
import os

from utils.utils4nc import preprocess_features
from utils.utils4nc import preprocess_adj

def _load_node_dataset(_dataset,dir_path = './data/'):
    """Load a node dataset.
    :param _dataset: Which dataset to load. Choose from "syn1", "syn2", "syn3" or "syn4"
    :returns: np.array
    """
    path = dir_path + _dataset + '/' + _dataset + '.pkl'
    with open(path, 'rb') as fin:
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix  = pkl.load(fin)
    labels = y_train
    labels[val_mask] = y_val[val_mask]
    labels[test_mask] = y_test[test_mask]

    return adj, features, labels, train_mask, val_mask, test_mask

def load_dataset(_dataset, shuffle = True):
    """High level function which loads the dataset
    by calling others spesifying in nodes or graphs.
    Keyword arguments:
    :param _dataset: Which dataset to load. Choose from "syn1", "syn2", "syn3", "syn4", "ba2" or "mutag"
    :param skip_preproccessing: Whether or not to convert the adjacency matrix to an edge matrix.
    :param shuffle: Should the returned dataset be shuffled or not.
    :returns: multiple np.arrays
    """
    print(f"Loading {_dataset} dataset...")
    adj, features, labels, train_mask, val_mask, test_mask = _load_node_dataset(_dataset)
    preprocessed_features = preprocess_features(features).astype('float32')
    graph = preprocess_adj(adj)[0].astype('int64').T
    labels = np.argmax(labels, axis=1)
    return graph, preprocessed_features, labels, train_mask, val_mask, test_mask

def _load_node_dataset_ground_truth(_dataset):
    """Load a the ground truth from a synthetic node dataset.
    Mutag is a large dataset and can thus take a while to load into memory.
    
    :param shuffle: Whether the data should be shuffled.
    :returns: np.array, np.array
    """
    dir_path = './data/'
    path = dir_path + _dataset + '/' + _dataset + '.pkl'
    with open(path, 'rb') as fin:
        adj, _, _, _, _, _, _, _, edge_label_matrix  = pkl.load(fin)
    graph = preprocess_adj(adj)[0].astype('int64').T
    labels = []
    for pair in graph.T:
        labels.append(edge_label_matrix[pair[0], pair[1]])
    labels = np.array(labels)
    return graph, labels

def load_dataset_ground_truth(_dataset, test_indices=None):
    """Load a the ground truth from a dataset.
    Optionally we can only request the indices needed for testing.
    
    :param test_indices: Only return the indices used by the PGExplaier paper.
    :returns: (np.array, np.array), np.array
    """
    if _dataset == "BAShapes" or _dataset == "BACommunity":
        graph, labels = _load_node_dataset_ground_truth(_dataset)
        if test_indices is None:
            return (graph, labels), range(400, 700, 5)
        else:
            all = range(400, 700, 1)
            filtered = [i for i in all if i in test_indices]
            return (graph, labels), filtered
    if _dataset == "TreeCycles":
        graph, labels = _load_node_dataset_ground_truth(_dataset)
        if test_indices is None:
            return (graph, labels), range(511,871,6)
        else:
            all = range(511, 871, 1)
            filtered = [i for i in all if i in test_indices]
            return (graph, labels), filtered
    if _dataset == "TreeGrid":
        graph, labels = _load_node_dataset_ground_truth(_dataset)
        if test_indices is None:
            return (graph, labels), range(511,800,1)
        else:
            all = range(511, 800, 1)
            filtered = [i for i in all if i in test_indices]
            return (graph, labels), filtered