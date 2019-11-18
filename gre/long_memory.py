""" Long Memory Components
"""
import torch.nn as nn


class NodeParams(nn.Module):
    """ Contains model parameters of size n_nodes x ...
    """
    def __init__(self):
        super(NodeParams, self).__init__()

    def forward(self, *args, **kwargs):
        pass

    def get(self, nodes, sparse_size):
        """ [Param] nodes (tensor): 2 x N where col = (idx, v) sorted
            [Return] params (sparse tensor): (idx, v) -> ...
        """
        raise NotImplementedError


class ETypeParams(nn.Module):
    """ Contains model parameters of size n_etypes x ...
    """
    def __init__(self):
        super(ETypeParams, self).__init__()

    def forward(self, *args, **kwargs):
        pass

    def get(self, edges, sparse_size):
        """ [Param] edges (tensor): 5 x N where col = (idx, v1, v2, et, di) sorted by (idx, v1, v2)
            [Return] params (sparse tensor): (idx, v1, v2, et, di) -> ...
        """
        raise NotImplementedError


class GlobalParams(nn.Module):
    """ Contains model parameters of size ...
    """
    def __init__(self):
        super(GlobalParams, self).__init__()

    def forward(self, *args, **kwargs):
        pass

    def get(self):
        """ [Return] params (tensor): ...
        """
        raise NotImplementedError


class LongMemory(nn.Module):
    """ Contains model parameters on the node-, etype-, and global-level
    """
    def __init__(self, node_params, etype_params, global_params):
        super(LongMemory, self).__init__()
        self.node_params = node_params
        self.etype_params = etype_params
        self.global_params = global_params

    def forward(self, *args, **kwargs):
        raise NotImplementedError
