"""
Graph Reasoning Engine
"""
import numpy as np

import torch
import torch.nn as nn

import gre.fast_numpy as fnp
import gre.sparse_ops as sop
from gre.long_memory import LongMemory
from gre.short_memory import ShortMemory
from gre.attention_tracker import AttentionTracker


class Engine(nn.Module):
    def __init__(self, hparams, context, graph_accessor, attention, transition,
                 node_params=None, etype_params=None, global_params=None,
                 node_states=None, edge_states=None, global_states=None):
        super(Engine, self).__init__()
        self.hparams = hparams
        self.context = context
        self.graph_accessor = graph_accessor
        self.attn_tracker = AttentionTracker(attention, transition)
        self.long_memory = LongMemory(node_params, etype_params, global_params)
        self.short_memory = ShortMemory(node_states, edge_states, global_states)

    def initialize_attention(self, *args, **kwargs):
        self.attn_tracker.initialize(self, *args, **kwargs)

    def initialize_short_memory(self, *args, **kwargs):
        self.short_memory.initialize(self, *args, **kwargs)

    def forward(self, *inputs, **kwargs):
        """ Performs one-step reasoning
        """
        attn = self.attn_tracker.get_attn()  # (sparse tensor) (idx, v) -> 1

        attd_from_nodes = self.graph_accessor.get_attd_from_nodes(self.hparams, self.context, attn)  # (2D numpy) (idx, v1) sorted by (idx, v1)
        attd_over_edges = self.graph_accessor.get_attd_over_edges(self.hparams, self.context, attd_from_nodes)  # (2D numpy) (idx, e, v1, v2) sorted by (idx, v1, v2)

        transition = self._attend(attd_over_edges)  # (sparse tensor) (idx, v1, v2) -> 1
        attn = self._update_attn(attn, transition)  # (sparse tensor) (idx, v) -> 1

        attd_to_nodes = self.graph_accessor.get_attd_to_nodes(attn, self.hparams. self.context)  # (2D numpy) (idx, v2) sorted by (idx, v2)
        prop_over_edges = self.graph_accessor.get_prop_over_edges(attd_over_edges, attd_to_nodes,
                                                                  self.hparams, self.context)  # (2D numpy) (idx, e, v1, v2) sorted by (idx, v1, v2)

        message = self._propagate(prop_over_edges)  # (sparse tensor) (idx, v1, v2) -> ...
        self._update_edges(message)

        message_aggr = self._aggregate(message)
        self._update_nodes(attd_to_nodes, message_aggr)

    def _attend(self, attd_over_edges):
        """ [Param] attd_over_edges (2D numpy): (idx, e, v1, v2) sorted by (idx, v1, v2)
        """
        from_nodes, to_nodes, from_sp_size, to_sp_size = self.get_from_and_to_nodes(attd_over_edges)  # (idx, v1), (idx, v2)
        via_edges, via_edges_et, via_sp_size, via_et_sp_size = self.get_via_edges_and_etypes(attd_over_edges)  # (idx, v1, v2), (idx, v1, v2, et, di)

        from_states, to_states = self.get_from_and_to_states(self.short_memory.node_states, from_nodes, to_nodes,
                                                             from_sp_size, to_sp_size)  # (idx, v1) -> ..., (idx, v2) -> ...
        via_states = self.get_via_states(self.short_memory.edge_states, via_edges, via_sp_size)  # (idx, v1, v2) -> ...
        g_states = self.get_global_states(self.short_memory.global_states)  # (idx,) -> ...

        from_params, to_params = self.get_from_and_to_params(self.long_memory.node_params, from_nodes, to_nodes,
                                                             from_sp_size, to_sp_size)  # (idx, v1) -> ..., (idx, v2) -> ...
        via_params = self.get_via_params(self.long_memory.etype_params, via_edges_et, via_et_sp_size)  # (idx, v1, v2, et, di) -> ...
        g_params = self.get_global_params(self.long_memory.global_params)  # ...

        transition = self.compute_transition(via_edges_et,
                                             from_nd_states, to_nd_states, over_e_states, global_state,
                                             from_nd_params, to_nd_params, over_e_params, global_params)
        return transition

    def _propagate(self, prop_over_edges):
        ''' [Param] prop_over_edges (2D numpy): (idx, e, v1, v2) sorted by (idx, e, v1, v2)
        '''
        from_nd_states, to_nd_states = self.get_from_and_to_node_states(prop_over_edges, self.short_memory.node_states, op='propagete')  # (idx, v1) -> ..., (idx, v2) -> ...
        over_e_states = self.get_over_edge_states(prop_over_edges, self.short_memory.edge_states, op='propagete')  # (idx, v1, v2) -> ...
        global_state = self.short_memory.global_states  # B x ...

        from_nd_params, to_nd_params = self.get_from_and_to_node_params(prop_over_edges, self.long_memory.node_params, op='propagete')  # (v1,) -> ..., (v2,) -> ...
        over_e_params = self.get_over_edge_params(prop_over_edges, self.long_memory.etype_params,
                                                  self.graph_accessor.info_full_edges, self.graph_accessor.info_temp_edges, op='propagete')
        global_params = self.long_memory.global_params  # ...

        transition = self.compute_message(prop_over_edges[:, [0, 2, 3]],
                                          from_nd_states, to_nd_states, over_e_states, global_state,
                                          from_nd_params, to_nd_params, over_e_params, global_params)
        return transition

    def get_from_and_to_nodes(self, over_edges):
        """ [Param] over_edges (2D numpy): (idx, e, v1, v2) sorted by (idx, v1, v2)
            [Return] from_nodes (tensor): 2 x N where col = (idx, v1) sorted
            [Return] to_nodes (tensor): 2 x N where col = (idx, v2) sorted
        """
        from_nodes = fnp.s_unique(over_edges[:, [0, 2]])
        from_nodes = sop.to_indices(from_nodes, device=self.context['device'])  # (idx, v1) sorted
        to_nodes = fnp.s_unique(fnp.sort(over_edges[:, [0, 3]]))
        to_nodes = sop.to_indices(to_nodes, device=self.context['device'])  # (idx, v2) sorted
        batch_size = over_edges[:, 0].max() + 1
        n_nodes = self.graph_accessor.n_nodes
        from_sp_size = to_sp_size = (batch_size, n_nodes)
        return from_nodes, to_nodes, from_sp_size, to_sp_size

    def get_via_edges_and_etypes(self, over_edges):
        """ [Param] over_edges (2D numpy): (idx, e, v1, v2) sorted by (idx, v1, v2)
            [Return] via_edges (tensor): 3 x N where col = (idx, v1, v2) sorted
            [Return] via_edges_et (tensor): 5 x N where col = (idx, v1, v2, et, di) sorted by (idx, v1, v2, et, di)
        """
        via_edges = fnp.sorted_unique(over_edges[:, [0, 2, 3]])
        via_edges = sop.to_indices(via_edges, device=self.context['device'])
        via_edges_et = self.graph_accessor.get_etypes(over_edges[:, 1])  # (et, di)
        via_edges_et = np.concatenate([over_edges[:, [0, 2, 3]], via_edges_et], 1)  # (idx, v1, v2, et, di)
        via_edges_et = fnp.sorted_unique(fnp.sort(via_edges_et))
        via_edges_et = sop.to_indices(via_edges_et, device=self.context['device'])
        batch_size = over_edges[:, 0].max() + 1
        n_nodes = self.graph_accessor.n_nodes
        n_etypes = self.graph_accessor.n_etypes
        via_sp_size = (batch_size, n_nodes, n_nodes)
        via_et_sp_size = (batch_size, n_nodes, n_nodes, n_etypes, 2)
        return via_edges, via_edges_et, via_sp_size, via_et_sp_size

    @staticmethod
    def get_from_and_to_states(node_states, from_nodes, to_nodes, from_sp_size, to_sp_size):
        """ [Param] node_states (module containing sparse tensor): (idx, v) -> ...
            [Param] from_nodes (tensor): 2 x N where col = (idx, v1) sorted
            [Param] to_nodes (tensor): 2 x N where col = (idx, v2) sorted
            [Return] from_states (sparse tensor): (idx, v1) -> ...
            [Return] to_states (sparse_tensor): (idx, v2) -> ...
        """
        from_states, to_states = None, None
        if node_states is not None:
            from_states = node_states.get(from_nodes, from_sp_size)  # (idx, v1) -> ...
            to_states = node_states.get(to_nodes, to_sp_size)  # (idx, v2) -> ...
        return from_states, to_states

    @staticmethod
    def get_via_states(edge_states, via_edges, via_sp_size):
        """ [Param] edge_states (module containing sparse tensor): (idx, v1, v2) -> ...
            [Param] via_edges (tensor): 3 x N where col = (idx, v1, v2) sorted
            [Return] via_states (sparse tensor): (idx, v1, v2) -> ...
        """
        via_states = None
        if via_states is not None:
            via_states = edge_states.get(via_edges, via_sp_size)  # (idx, v1, v2) -> ...
        return via_states

    @staticmethod
    def get_global_states(global_states):
        """ [Param] global_states (module containing tensor): B x ...
            [Return] g_states (sparse tensor): (idx,) -> ...
        """
        g_states = None
        if global_states is not None:
            g_states = global_states.get(return_sparse=True)
        return g_states

    @staticmethod
    def get_from_and_to_params(node_params, from_nodes, to_nodes, from_sp_size, to_sp_size):
        """ [Param] node_params (module containing tensor): n_nodes x ...
            [Param] from_nodes (tensor): 2 x N where col = (idx, v1) sorted
            [Param] to_nodes (tensor): 2 x N where col = (idx, v2) sorted
            [Return] from_params (sparse tensor): (idx, v1) -> ...
            [Return] to_params (sparse tensor): (idx, v2) -> ...
        """
        from_params, to_params = None, None
        if node_params is not None:
            from_params = node_params.get(from_nodes, from_sp_size)  # (idx, v1) -> ...
            to_params = node_params.get(to_nodes, to_sp_size)  # (idx, v2) -> ...
        return from_params, to_params

    @staticmethod
    def get_via_params(etype_params, via_edges_et, via_et_sp_size):
        """ [Param] etype_params (module containing tensor): n_etypes x ...
            [Param] via_edges_et (tensor): 5 x N where col = (idx, v1, v2, et, di) sorted by (idx, v1, v2, et, di)
            [Return] via_params (sparse tensor): (idx, v1, v2, et, di) -> ...
        """
        via_params = None
        if etype_params is not None:
            via_params = etype_params.get(via_edges_et, via_et_sp_size)  # (idx, v1, v2, et, di) -> ...
        return via_params

    @staticmethod
    def get_global_params(global_params):
        """ [Param] global_params (module containing tensor): ...
            [Return] g_params (tensor): ...
        """
        g_params = None
        if global_params is not None:
            g_params = global_params.get()
        return g_params

    def compute_transition(self, via_edge, via_edges_et, via_sp_size, via_et_sp_size,
                           from_states, to_states, via_states, g_states,
                           from_params, to_params, via_params, g_params):
        """ [Param] via_edges_et (tensor): 5 x N where col = (idx, v1, v2, et, di) sorted by (idx, v1, v2, et, di)
            [Param] from_states (sparse tensor): (idx, v1) -> ...
            [Param] to_states (sparse_tensor): (idx, v2) -> ...
            [Param] via_states (sparse tensor): (idx, v1, v2) -> ...
            [Param] g_states (sparse tensor): (idx,) -> ...
            [Param] from_params (sparse tensor): (idx, v1) -> ...
            [Param] to_params (sparse tensor): (idx, v2) -> ...
            [Param] via_params (sparse tensor): (idx, v1, v2, et, di) -> ...
            [Param] g_params (tensor): ...
            [Return] transition (sparse tensor): (idx, v1, v2) -> 1
        """
        if from_states is not None:
            from_states = sop.sparse_broadcast(from_states, via_edges_et, n_dims=2, sparse_size=via_et_sp_size)  # (idx, v1, v2, et, di) -> ...
        if to_states is not None:
            to_states = sop.sparse_broadcast(to_states, via_edges_et, dim=[0, 2], sparse_size=via_et_sp_size)  # (idx, v1, v2, et, di) -> ...
        if via_states is not None:
            via_states = sop.sparse_broadcast(via_states, via_edges_et, n_dims=3, sparse_size=via_et_sp_size)  # (idx, v1, v2, et, di) -> ...
        if g_states is not None:
            g_states = sop.sparse_broadcast(g_states, via_edges_et, n_dims=1, sparse_size=via_et_sp_size)  # (idx, v1, v2, et, di) -> ...
        if from_params is not None:
            from_params = sop.sparse_broadcast(from_params, via_edges_et, n_dims=2, sparse_size=via_et_sp_size)  # (idx, v1, v2, et, di) -> ...
        if to_params is not None:
            to_params = sop.sparse_broadcast(to_params, via_edges_et, dim=[0, 2], sparse_size=via_et_sp_size)  # (idx, v1, v2, et, di) -> ...
        if g_params is not None:
            g_params = sop.sparse_fill(via_edges_et, g_params, sparse_size=via_et_sp_size)  # (idx, v1, v2, et, di) -> ...

        transition_logits = self.compute_transition_logits(from_states, to_states, via_states, g_states,
                                                           from_params, to_params, via_params, g_params)

        transition = transition_logits  # Todo: softmax
        return transition

    def compute_transition_logits(self,
                                  from_states, to_states, via_states, g_states,
                                  from_params, to_params, via_params, g_params):
        """ [Return] logits (sparse tensor): (idx, v1, v2, et, di) -> 1
        """
        raise NotImplementedError

    def _update_attn(self, attn, transition):
        ''' [Param] attn (sparse tensor) (idx, v) -> 1
            [Param] transition (sparse tensor): (idx, v1, v2) -> 1
        '''
        pass

    def compute_message(self, edges,
                        from_nd_states, to_nd_states, e_states, global_state,
                        from_nd_params, to_nd_params, e_params, global_params):
        ''' [Param] edges (2D numpy): (idx, v1, v2) sorted by (idx, v1, v2)
            [Param] from_nd_states (sparse tensor): (idx, v1) -> ...
            [Param] to_nd_states (sparse_tensor): (idx, v2) -> ...
            [Param] e_states (sparse tensor): (idx, v1, v2) -> ...
            [Param] global_states (tensor): B x ...
            [Param] from_nd_params (sparse tensor): (v1,) -> ...
            [Param] to_nd_params (sparse tensor): (v2,) -> ...
            [Param] e_params (sparse tensor): (v1, v2) -> ...
            [Param] global_params (tensor): ...
            [Return] message (sparse tensor): (idx, v1, v2) -> ...
        '''
        raise NotImplementedError
