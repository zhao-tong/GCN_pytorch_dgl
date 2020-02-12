import os
import sys
import time
import pickle
import warnings
import numpy as np
import networkx as nx
import scipy.sparse as sp
import dgl
from dgl import DGLGraph
import torch
from collections import defaultdict

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

class DataLoader():
    def __init__(self, args):
        self.args = args
        self.dataset = args.dataset
        self.G, self.labels, self.features, self.adj_mat = None, None, None, None
        self.train_nid = None

        self.load_data(self.dataset)

    def load_data(self, dataset):
        """
        Modified from Kipf's code: https://github.com/tkipf/gcn
        Loads input data from gcn/data directory
        ind.dataset.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
        ind.dataset.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
        ind.dataset.allx => the feature vectors of both labeled and unlabeled training instances
            (a superset of ind.dataset.x) as scipy.sparse.csr.csr_matrix object;
        ind.dataset.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
        ind.dataset.ty => the one-hot labels of the test instances as numpy.ndarray object;
        ind.dataset.ally => the labels for instances in ind.dataset.allx as numpy.ndarray object;
        ind.dataset.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
            object;
        ind.dataset.test.index => the indices of test instances in graph, for the inductive setting as list object.
        All objects above must be saved using python pickle module.
        :param dataset: Dataset name
        """
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for n in names:
            with open(f"{CUR_DIR}/data/ind.{dataset}.{n}", 'rb') as f:
                objects.append(pickle.load(f, encoding='latin1'))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = []
        for line in open(f"{CUR_DIR}/data/ind.{dataset}.test.index"):
            test_idx_reorder.append(int(line.strip()))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).toarray()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        self.features = torch.FloatTensor(features)

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj = sp.coo_matrix(adj)
        adj.setdiag(1)
        adj = sp.csr_matrix(adj)
        self.adj_mat = adj
        self.G = DGLGraph(adj)

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        self.n_class = labels.shape[1]
        self.labels = torch.LongTensor(np.argmax(labels, axis=1))

        self.train_nid = np.arange(len(y))
        self.val_nid = np.arange(len(y), len(y)+500)
        self.test_nid = test_idx_range
