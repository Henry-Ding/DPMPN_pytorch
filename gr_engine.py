'''
Graph Reasoning Engine
'''

import torch
import torch.nn as nn


class NodeParams(nn.Module):
    ''' Contains model parameters of size n_nodes x ...
    '''
    def __init__(self):
        super(NodeParams, self).__init__()


class ETypeParams(nn.Module):
    ''' Contains model parameters of size n_etypes x ...
    '''
    def __init__(self):
        super(ETypeParams, self).__init__()


class GlobalParams(nn.Module):
    ''' Contains model parameters of size ...
    '''
    def __init__(self):
        super(GlobalParams, self).__init__()


class LongMemory(nn.Module):
    def __init__(self, node_params, etype_params, global_params):
        super(LongMemory, self).__init__()
        self.node_params = node_params  # contains model parameters of size n_nodes x ...
        self.etype_params = etype_params  # contains model parameters of size n_etypes x ...
        self.global_params = global_params  # contains model parameters of size ...


class NodeStates(nn.Module):
    ''' Contains intermediate sparse tensors of form (eg, v) -> ...
    '''
    def __init__(self):
        super(NodeStates, self).__init__()

    def initialize(self):
        raise NotImplementedError


class EdgeStates(nn.Module):
    ''' Contains intermediate sparse tensors of form (eg, v1, v2) -> ...
    '''
    def __init__(self):
        super(EdgeStates, self).__init__()

    def initialize(self):
        raise NotImplementedError


class GlobalStates(nn.Module):
    ''' Contains intermediate tensors of size B x ...
    '''
    def __init__(self):
        super(GlobalStates, self).__init__()

    def initialize(self):
        raise NotImplementedError


class ShortMemory(nn.Module):
    def __init__(self, node_states, edge_states, global_states):
        super(ShortMemory, self).__init__()
        self.node_states = node_states  # contains intermediate sparse tensors of form (eg, v) -> ...
        self.edge_states = edge_states  # contains intermediate sparse tensors of form (eg, v1, v2) -> ...
        self.global_states = global_states  # contains intermediate tensors of size B x ...

    def initialize(self):
        self.nodes_mem and self.nodes_mem.initialize()
        self.edges_mem and self.edges_mem.initialize()
        self.global_mem and self.global_mem.initialize()


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def initialize(self):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError


class AttnTracker(nn.Module):
    def __init__(self, attention):
        super(AttnTracker, self).__init__()
        self.tracker = []
        self.attention = attention  # contains current attention sparse tensors of form (eg, v) -> 1

    def initialize(self):
        self.attention.initialize()
        self.tracker = [self.attention.copy()]

    def get_attn(self):
        # Get attention at the current step
        return self.attention


class GraphAccessor(object):
    def __init__(self, edge_data):
        ''' [Param] edge_data: (v1, v2) -> (et, di) where et in {0, 1, 2, ...} and di in {-1, 1}
        '''
        self.info_full_edges = self._make_full_edges(edge_data)  # (2D numpy) (e, v1, v2, et, di) sorted by (v1, v2)
        self.info_temp_edges = {-1: (-1, 1)}  # (dict) e -> (et, di), selfloop (e = -1)

        self.used_edges = self.info_full_edges[:, :3]  # (2D numpy) (e, v1, v2) sorted by (v1, v2)
        self.edges = None  # (2D numpy) (eg, e, v1, v2) sorted by (eg, v1, v2)

    def _make_full_edges(self, edge_data):
        ''' [Return] full_edges ﻿(2D numpy): (e, v1, v2, et, di) sorted by (v1, v2), consecutively increasing e starting with 0
        '''
        # Todo
        pass

    def reset_graph(self, edge_data):
        self.info_full_edges = self._make_full_edges(edge_data)

    def use_edges(self, edge_ids):
        ''' [Param] edge_ids (list) sorted
        '''
        self.used_edges = self.info_full_edges[edge_ids][:, :3]

    def get_topk_nodes(self, attn, k):
        ''' [Param] attn (sparse tensor): (eg, v) -> 1
            [Return] topk_nodes (2D numpy): (eg, v) sorted by (eg, v)
        '''
        pass

    def get_out_edges(self, nodes, edges):
        ''' [Param] nodes (2D numpy): (eg, v) sorted by (eg, v)
            [Param] edges (eg-invariant) (2D numpy): (e, v1, v2) sorted by (v1, v2)
            [Return] out_edges (2D numpy): (eg, e, v1, v2) sorted by (eg, v1, v2)
        '''
        pass

    def sample_edges(self, edges, k, per_v1=True):
        ''' [Param] edges (eg-dependent) (2D numpy): (eg, e, v1, v2) sorted by (eg, v1, v2)
            [Return] sampled_edges (2D numpy): (eg, e, v1, v2) sorted by (eg, v1, v2)
        '''
        pass

    def add_selfloop(self, edges):
        ''' [Param] edges (eg-dependent) (2D numpy): (eg, e, v1, v2) sorted by (eg, v1, v2)
            [Return] aug_edges (2D numpy): (eg, e, v1, v2) sorted by (eg, v1, v2) with selfloop edges (eg, -1, v1, v1)
        '''
        pass

    def get_attd_from_nodes(self, attn, hparams, context):
        ''' [Param] attn (sparse tensor): (eg, v) -> 1
            [Returm] attd_from_nodes (2D numpy): (eg, v1) sorted by (eg, v1)
        '''
        raise NotImplementedError

    def get_attd_to_nodes(self, attn, hparams, context):
        ''' [Param] attn (sparse tensor): (eg, v) -> 1
            [Returm] attd_to_nodes (2D numpy): (eg, v2) sorted by (eg, v2)
        '''
        raise NotImplementedError

    def get_attd_over_edges(self, attd_from_nodes, used_edges, hparams, context):
        ''' [Param] attd_from_nodes (2D numpy): (eg, v1) sorted by (eg, v1)
            [Param] used_edges (2D numpy): (e, v1, v2) sorted by (v1, v2)
            [Return] attd_over_edges (2D numpy): (eg, e, v1, v2) sorted by (eg, e, v1, v2)
        '''
        raise NotImplementedError

    def get_prop_over_edges(self, attd_over_edges, attd_to_nodes, hparams, context):
        ''' [Param] attd_over_edges (2D numpy): (eg, e, v1, v2) sorted by (eg, e, v1, v2)
            [Param] attd_to_nodes (2D numpy): (eg, v2) sorted by (eg, v2)
            [Return] prop_over_edges (2D numpy): (eg, e, v1, v2) sorted by (eg, e, v1, v2)
        '''
        raise NotImplementedError


class Engine(nn.Module):
    def __init__(self, hparams, graph_accessor, attention,
                 node_params=None, etype_params=None, global_params=None,
                 node_states=None, edge_states=None, global_states=None):
        super(Engine, self).__init__()
        self.hparams = hparams
        self.context = dict()
        self.graph_accessor = graph_accessor
        self.attn_tracker = AttnTracker(attention)
        self.long_memory = LongMemory(node_params, etype_params, global_params)
        self.short_memory = ShortMemory(node_states, edge_states, global_states)

    def initialize(self):
        self.attn_tracker.initialize()
        self.short_memory.initialize()

    def forward(self, *inputs, **kwargs):
        attn = self.attn_tracker.get_attn()  # (sparse tensor) (eg, v) -> 1

        attd_from_nodes = self.graph_accessor.get_attd_from_nodes(attn, self.hparams, self.context)  # (2D numpy) (eg, v1) sorted by (eg, v1)
        attd_over_edges = self.graph_accessor.get_attd_over_edges(attd_from_nodes, self.graph_accessor.used_edges,
                                                                  self.hparams, self.context)  # (2D numpy) (eg, e, v1, v2) sorted by (eg, v1, v2)

        transition = self._attend(attd_over_edges)  # (sparse tensor) (eg, v1, v2) -> 1
        attn = self._update_attn(attn, transition)  # (sparse tensor) (eg, v) -> 1

        attd_to_nodes = self.graph_accessor.get_attd_to_nodes(attn, self.hparams. self.context)  # (2D numpy) (eg, v2) sorted by (eg, v2)
        prop_over_edges = self.graph_accessor.get_prop_over_edges(attd_over_edges, attd_to_nodes,
                                                                  self.hparams, self.context)  # (2D numpy) (eg, e, v1, v2) sorted by (eg, v1, v2)

        message = self._propagate(prop_over_edges)  # (sparse tensor) (eg, v1, v2) -> ...
        self._update_edges(message)

        message_aggr = self._aggregate(message)
        self._update_nodes(attd_to_nodes, message_aggr)

    def _attend(self, attd_over_edges):
        ''' [Param] attd_over_edges (2D numpy): (eg, e, v1, v2) sorted by (eg, e, v1, v2)
        '''
        from_nd_states, to_nd_states = self.get_from_and_to_node_states(attd_over_edges, self.short_memory.node_states, op='attend')  # (eg, v1) -> ..., (eg, v2) -> ...
        over_e_states = self.get_over_edge_states(attd_over_edges, self.short_memory.edge_states, op='attend')  # (eg, v1, v2) -> ...
        global_state = self.short_memory.global_states  # B x ...

        from_nd_params, to_nd_params = self.get_from_and_to_node_params(attd_over_edges, self.long_memory.node_params, op='attend')  # (v1,) -> ..., (v2,) -> ...
        over_e_params = self.get_over_edge_params(attd_over_edges, self.long_memory.etype_params,
                                                  self.graph_accessor.info_full_edges, self.graph_accessor.info_temp_edges, op='attend')
        global_params = self.long_memory.global_params  # ...

        transition = self.compute_transition(attd_over_edges[:, [0, 2, 3]],
                                             from_nd_states, to_nd_states, over_e_states, global_state,
                                             from_nd_params, to_nd_params, over_e_params, global_params)
        return transition

    def _propagate(self, prop_over_edges):
        ''' [Param] prop_over_edges (2D numpy): (eg, e, v1, v2) sorted by (eg, e, v1, v2)
        '''
        from_nd_states, to_nd_states = self.get_from_and_to_node_states(prop_over_edges, self.short_memory.node_states, op='propagete')  # (eg, v1) -> ..., (eg, v2) -> ...
        over_e_states = self.get_over_edge_states(prop_over_edges, self.short_memory.edge_states, op='propagete')  # (eg, v1, v2) -> ...
        global_state = self.short_memory.global_states  # B x ...

        from_nd_params, to_nd_params = self.get_from_and_to_node_params(prop_over_edges, self.long_memory.node_params, op='propagete')  # (v1,) -> ..., (v2,) -> ...
        over_e_params = self.get_over_edge_params(prop_over_edges, self.long_memory.etype_params,
                                                  self.graph_accessor.info_full_edges, self.graph_accessor.info_temp_edges, op='propagete')
        global_params = self.long_memory.global_params  # ...

        transition = self.compute_message(prop_over_edges[:, [0, 2, 3]],
                                          from_nd_states, to_nd_states, over_e_states, global_state,
                                          from_nd_params, to_nd_params, over_e_params, global_params)
        return transition

    def get_from_and_to_node_states(self, attd_over_edges, sm_node_states, op='attend'):
        ''' [Param] attd_over_edges (2D numpy): (eg, e, v1, v2) sorted by (eg, e, v1, v2)
            [Param] sm_node_states (sparse tensor): (eg, v) -> ...
            [Param] op: 'attend' or 'propagate'
            [Return] from_nd_states (sparse tensor): (eg, v1) -> ...
            [Return] to_nd_states (sparse_tensor): (eg, v2) -> ...
        '''
        raise NotImplementedError

    def get_over_edge_states(self, attd_over_edges, sm_edge_states, op='attend'):
        ''' [Param] attd_over_edges (2D numpy): (eg, e, v1, v2) sorted by (eg, e, v1, v2)
            [Param] sm_edge_states (sparse tensor): (eg, v1, v2) -> ...
            [Return] over_e_states (sparse tensor): (eg, v1, v2) -> ...
        '''
        raise NotImplementedError

    def get_from_and_to_node_params(self, attd_over_edges, lm_node_params, op='attend'):
        ''' [Param] attd_over_edges (2D numpy): (eg, e, v1, v2) sorted by (eg, e, v1, v2)
            [Param] lm_node_params (tensor): n_nodes x ...
            [Return] from_nd_params (sparse tensor): (v1,) -> ...
            [Return] to_nd_params (sparse tensor): (v2,) -> ...
        '''
        raise NotImplementedError

    def get_over_edge_params(self, attd_over_edges, lm_etype_params, info_full_edges, info_temp_edges, op='attend'):
        ''' [Param] attd_over_edges (2D numpy): (eg, e, v1, v2) sorted by (eg, e, v1, v2)
            [Param] lm_etype_params (tensor): n_etypes x ...
            [Return] over_e_params (sparse tensor): (v1, v2) -> ...
        '''
        raise NotImplementedError

    def compute_transition(self, edges,
                           from_nd_states, to_nd_states, e_states, global_state,
                           from_nd_params, to_nd_params, e_params, global_params):
        ''' [Param] edges (2D numpy): (eg, v1, v2) sorted by (eg, v1, v2)
            [Param] from_nd_states (sparse tensor): (eg, v1) -> ...
            [Param] to_nd_states (sparse_tensor): (eg, v2) -> ...
            [Param] e_states (sparse tensor): (eg, v1, v2) -> ...
            [Param] global_states (tensor): B x ...
            [Param] from_nd_params (sparse tensor): (v1,) -> ...
            [Param] to_nd_params (sparse tensor): (v2,) -> ...
            [Param] e_params (sparse tensor): (v1, v2) -> ...
            [Param] global_params (tensor): ...
            [Return] transition (sparse tensor): (eg, v1, v2) -> 1
        '''
        transition_logits = self.compute_transition_logits(edges,
                                                           from_nd_states, to_nd_states, e_states, global_state,
                                                           from_nd_params, to_nd_params, e_params, global_params)
        transition = transition_logits # Todo: softmax
        return transition

    def compute_transition_logits(self, edges,
                                  from_nd_states, to_nd_states, e_states, global_state,
                                  from_nd_params, to_nd_params, e_params, global_params):
        ''' [Param] edges (2D numpy): (eg, v1, v2) sorted by (eg, v1, v2)
            [Param] from_nd_states (sparse tensor): (eg, v1) -> ...
            [Param] to_nd_states (sparse_tensor): (eg, v2) -> ...
            [Param] e_states (sparse tensor): (eg, v1, v2) -> ...
            [Param] global_states (tensor): B x ...
            [Param] from_nd_params (sparse tensor): (v1,) -> ...
            [Param] to_nd_params (sparse tensor): (v2,) -> ...
            [Param] e_params (sparse tensor): (v1, v2) -> ...
            [Param] global_params (tensor): ...
            [Return] transition_logits (sparse tensor): (eg, v1, v2) -> 1
        '''
        raise  NotImplementedError

    def _update_attn(self, attn, transition):
        ''' [Param] attn (sparse tensor) (eg, v) -> 1
            [Param] transition (sparse tensor): (eg, v1, v2) -> 1
        '''
        pass

    def compute_message(self, edges,
                        from_nd_states, to_nd_states, e_states, global_state,
                        from_nd_params, to_nd_params, e_params, global_params):
        ''' [Param] edges (2D numpy): (eg, v1, v2) sorted by (eg, v1, v2)
            [Param] from_nd_states (sparse tensor): (eg, v1) -> ...
            [Param] to_nd_states (sparse_tensor): (eg, v2) -> ...
            [Param] e_states (sparse tensor): (eg, v1, v2) -> ...
            [Param] global_states (tensor): B x ...
            [Param] from_nd_params (sparse tensor): (v1,) -> ...
            [Param] to_nd_params (sparse tensor): (v2,) -> ...
            [Param] e_params (sparse tensor): (v1, v2) -> ...
            [Param] global_params (tensor): ...
            [Return] message (sparse tensor): (eg, v1, v2) -> ...
        '''
        raise NotImplementedError

    def