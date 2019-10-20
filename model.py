import torch
import torch.nn as nn

import gre


class L_NodesMem(nn.Module):
    def __init__(self, hparams, graph, context):
        super(L_NodesMem, self).__init__()
        self.embs = nn.Parameter(torch.randn(graph.n_nodes, hparams.n_dims, device=context.device) * 0.01)


class L_ETypesMem(nn.Module):
    def __init__(self, hparams, graph, context):
        super(L_ETypesMem, self).__init__()
        self.embs = nn.Parameter(torch.randn(graph.n_etypes, hparams.n_dims, device=context.device) * 0.01)


class KG(object):
    def __init__(self, dataset):
        self.edge_data = dataset.triples
        self.n_nodes = dataset.n_entities
        self.n_etypes = dataset.n_relations
        self.reverse_rel_dct = dataset.reverse_rel_dct

    # todo

    def


class Model(object):
    def __init__(self, hparams, kg):
        self.hparams = hparams
        self.graph = gre.Graph(kg.edge_data, kg.add_reverse, kg.add_selfloop)
