import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl import DGLGraph

class GCN(nn.Module):
    def __init__(self, G, dim_in, dim_h, dim_z, n_class, dropout):
        super(GCN, self).__init__()
        self.G = G
        self.dim_z = dim_z
        self.layer0 = GCN_layer(G, dim_in, dim_h, dropout)
        self.layer1 = GCN_layer(G, dim_h, dim_z, dropout)
        self.layer2 = GCN_layer(G, dim_z, n_class, dropout, act=False)

    def forward(self, features, norm):
        h = self.layer0(features, norm)
        h = self.layer1(h, norm)
        x = self.layer2(h, norm)
        return x

class GCN_layer(nn.Module):
    def __init__(self, G, dim_in, dim_out, dropout, act=True):
        super(GCN_layer, self).__init__()
        self.G = G
        self.act = act
        self.dropout = dropout
        self.weight = self.glorot_init(dim_in, dim_out)
        self.linear = nn.Linear(dim_in, dim_out, bias=False)
        if self.dropout:
            self.dropout = nn.Dropout(p=dropout)

    def glorot_init(self, input_dim, output_dim):
        init_range = np.sqrt(6.0/(input_dim + output_dim))
        initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
        return nn.Parameter(initial)

    def forward(self, h, norm):
        if self.dropout:
            h = self.dropout(h)
        h = h @ self.weight
        self.G.ndata['h'] = h * norm
        self.G.update_all(fn.copy_src(src='h', out='m'),
                          fn.sum(msg='m', out='h'))
        h = self.G.ndata.pop('h') * norm
        if self.act:
            h = F.relu(h)
        return h
