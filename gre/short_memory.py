""" Short Memory Components
"""
import torch.nn as nn


class NodeStates(nn.Module):
    """ Contains intermediate sparse tensors of form (idx, v) -> ...
    """
    def __init__(self):
        super(NodeStates, self).__init__()

    def forward(self, *args, **kwargs):
        pass

    def initialize(self, *args, **kwargs):
        raise NotImplementedError

    def get(self, nodes, sparse_size):
        """ [Param] nodes (tensor): 2 x N where col = (idx, v) sorted
            [Return] states (sparse tensor): (idx, v) -> ...
        """
        raise NotImplementedError


class EdgeStates(nn.Module):
    """ Contains intermediate sparse tensors of form (idx, v1, v2) -> ...
    """
    def __init__(self):
        super(EdgeStates, self).__init__()

    def forward(self, *args, **kwargs):
        pass

    def initialize(self, *args, **kwargs):
        raise NotImplementedError

    def get(self, edges, sparse_size):
        """ [Param] edges (tensor): 3 x N where col = (idx, v1, v2) sorted
            [Return] states (sparse tensor): (idx, v1, v2) -> ...
        """
        raise NotImplementedError


class GlobalStates(nn.Module):
    """ Contains intermediate tensors of size B x ...
    """
    def __init__(self):
        super(GlobalStates, self).__init__()

    def forward(self, *args, **kwargs):
        pass

    def initialize(self, *args, **kwargs):
        raise NotImplementedError

    def get(self, return_sparse=False):
        """ [Return] states (tensor): B x ... or (sparse tensor): (idx,) -> ...
        """
        raise NotImplementedError


class ShortMemory(nn.Module):
    """ Contains intermediate states
    """
    def __init__(self, node_states, edge_states, global_states):
        super(ShortMemory, self).__init__()
        self.node_states = node_states  # contains intermediate sparse tensors of form (eg, v) -> ...
        self.edge_states = edge_states  # contains intermediate sparse tensors of form (eg, v1, v2) -> ...
        self.global_states = global_states  # contains intermediate tensors of size B x ...

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def initialize(self, engine, *args, **kwargs):
        self.nodes_mem and self.nodes_mem.initialize(engine, *args, **kwargs)
        self.edges_mem and self.edges_mem.initialize(engine, *args, **kwargs)
        self.global_mem and self.global_mem.initialize(engine, *args, **kwargs)
