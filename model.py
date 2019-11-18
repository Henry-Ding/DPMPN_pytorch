import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import gre
import gre.sparse_ops as sop
import gre.consts as const


class NodeParams(gre.NodeParams):
    def __init__(self, n_nodes, n_dims, device):
        super(NodeParams, self).__init__()
        self.params = nn.Parameter(torch.randn(n_nodes, n_dims, device=device) * 0.01)  # n_nodes x n_dims
        self.fn_embed = nn.Sequential(nn.Linear(n_dims, n_dims*2), nn.ReLU(), nn.Linear(n_dims*2, n_dims))
        self.fn_query = nn.Sequential(nn.Linear(n_dims, n_dims*2), nn.ReLU(), nn.Linear(n_dims*2, n_dims))

    def get(self, nodes, sparse_size):
        """ [Param] nodes (tensor): 2 x N where col = (idx, v) sorted
            [Return] params (sparse tensor): (idx, v) -> n_dims
        """
        return self.get_embed(nodes, sparse_size)

    def get_embed(self, nodes, sparse_size):
        """ [Param] nodes (tensor): 2 x N where col = (idx, v) sorted
            [Return] embed (sparse tensor): (idx, v) -> n_dims
        """
        return sop.index_select_to_sparse(self.params, 0, nodes, 1, sparse_size=sparse_size, fn_values=self.fn_embed)

    def get_query(self, heads):
        """ [Param] heads (tensor): B
            [Return] query (tensor): B x n_dims
        """
        return self.fn_query(torch.index_select(self.params, 0, heads))  # B x n_dims


class ETypeParams(gre.ETypeParams):
    def __init__(self, n_etypes, n_dims, device):
        super(ETypeParams, self).__init__()
        self.params_et = nn.Parameter(torch.randn(n_etypes, n_dims, device=device) * 0.01)  # n_etypes x n_dims
        self.params_di = nn.Parameter(torch.randn(2, n_dims, device=device) * 0.01)  # 2 x n_dims
        self.fn_embed = nn.Sequential(nn.Linear(n_dims*2, n_dims*2), nn.ReLU(), nn.Linear(n_dims*2, n_dims))
        self.fn_query = nn.Sequential(nn.Linear(n_dims*2, n_dims*2), nn.ReLU(), nn.Linear(n_dims*2, n_dims))

    def get(self, edges, sparse_size):
        """ [Param] edges (tensor): 5 x N where col = (idx, v1, v2, et, di) sorted by (idx, v1, v2)
            [Return] embed (sparse tensor): (idx, v1, v2, et, di) -> n_dims
        """
        return self.get_embed(edges, sparse_size)

    def get_embed(self, edges, sparse_size):
        """ [Param] edges (tensor): 5 x N where col = (idx, v1, v2, et, di) sorted by (idx, v1, v2)
            [Return] embed (sparse tensor): (idx, v1, v2, et, di) -> n_dims
        """
        return sop.index_select_to_sparse([self.params_et, self.params_di], 0, edges, [3, 4],
                                          sparse_size=sparse_size, fn_values=self.fn_embed)

    def get_query(self, rels, di):
        """ [Param] rels (tensor): B
            [Param] di (tensor): B
            [Return] query (tensor): B x n_dims
        """
        return self.fn_query(torch.cat([torch.index_select(self.params_et, 0, rels),
                                        torch.index_select(self.params_di, 0, di)], 1))  # B x n_dims


class NodeStates(gre.NodeStates):
    def __init__(self, batch_size, n_nodes, n_dims, device):
        super(NodeStates, self).__init__()
        self.states = sop.none_to_sparse((batch_size, n_nodes), (n_dims,), device=device)  # (sparse tensor) (idx, v) -> n_dims
        self.update_net = None  # todo

    def get(self, nodes, sparse_size):
        """ [Param] nodes (tensor): 2 x N where col = (idx, v) sorted
            [Return] states (sparse tensor): (idx, v) -> ...
        """
        return sop.sparse_index(self.states, nodes, sparse_size=sparse_size)

    def initialize(self, engine, heads):
        """ [Param] heads (tensor): B
        """
        idx_heads = sop.idx_indices(heads)  # 2 x B where col = (idx, v)
        self.states = self._update(idx_heads, engine.long_memory, engine.short_memory.global_states)  # (idx, v) -> n_dims

    def update(self, engine, nodes, message_aggr):
        # todo
        pass

    def _update(self, updated_nodes, long_memory, global_states, message_aggr=None):
        """ [Param] updated_nodes (tensor): 2 x B where col = (idx, v) sorted, only including the nodes to be updated
            [Param] message_aggr (sparse tensor): (idx, v) -> n_dims
            [Return] new_node_states (sparse tensors): (idx, v) -> n_dims
        """
        sparse_size = tuple(self.states.size()[:2])
        dense_size = tuple(self.states.size()[2])
        node_states = sop.sparse_index(self.states, updated_nodes)  # (sparse tensor) (idx, v) -> n_dims
        node_embeds = long_memory.node_params.get_embed(updated_nodes)  # (sparse tensor): (idx, v) -> n_dims
        query_heads = sop.index_select_to_sparse(global_states.query_heads, 0, updated_nodes, 0, sparse_size=sparse_size)  # (sparse tensor): (idx, v) -> n_dims
        query_rels = sop.index_select_to_sparse(global_states.query_rels, 0, updated_nodes, 0, sparse_size)  # (sparse tensor): (idx, v) -> n_dims
        if message_aggr is None:
            message_aggr = sop.none_to_sparse(sparse_size, dense_size, device=self.states.device)
        message_aggr = sop.sparse_index(message_aggr, updated_nodes)  # (sparse tensor): (idx, v) -> n_dims

        new_node_states = self.update_net(node_states, node_embeds, query_heads, query_rels, message_aggr)  # (sparse tensor) (idx, v) -> n_dims
        return new_node_states


class GlobalStates(gre.GlobalStates):
    def __init__(self):
        super(GlobalStates, self).__init__()
        self.query_heads = None  # B x n_dims
        self.query_rels = None

    def get(self, return_sparse=False):
        """ [Return] query_states (tensor): B x ... or (sparse tensor): (idx,) -> ...
        """
        query_states = torch.cat([self.query_heads, self.query_rels], 1)
        if return_sparse:
            batch_size = query_states.size(0)
            indices = torch.arange(batch_size).to(device=query_states.device)
            query_states = torch.sparse_coo_tensor(indices, query_states, size=query_states.size()).coalesce()
        return query_states

    def initialize(self, long_memory, heads, rels):
        """ [Param] heads (tensor): B
            [Param] rels (tensor): B
        """
        self.query_heads = long_memory.node_params.get_query(heads)  # B x n_dims
        self.query_rels = long_memory.etype_params.get_query(rels)  # B x n_dims


class Attention(gre.Attention):
    def __init__(self, n_nodes):
        super(Attention, self).__init__()
        self.n_nodes = n_nodes
        self.attn = None  # (sparse tensor): (idx, v) -> 1

    def get(self):
        return self.attn

    def initialize(self, heads):
        """ [Param] heads (tensor): B where elem = v
        """
        idx_heads = sop.idx_indices(heads)  # 2 x B where col = (idx, v)
        batch_size = idx_heads.size(1)
        self.attn = sop.sparse_one_hot(idx_heads, sparse_size=(batch_size, self.n_nodes))  # (idx, v) -> 1


class Transition(gre.Transition):
    def __init__(self):
        super(Transition, self).__init__()
        self.trans = None

    def get(self):
        return self.trans


class GraphAccessor(gre.GraphAccessor):
    def get_attd_to_nodes(self, hparams, context, attn):
        pass

    def get_attd_from_nodes(self, hparams, context, attn):
        from_nodes = self.get_topk_nodes_pi(attn, hparams.max_attd_from, threshold=1e-8)
        return from_nodes

    def get_attd_over_edges(self, attd_from_nodes, hparams, context):
        nei_edges = self.get_out_edges_pi(attd_from_nodes, self.edges, removed_edge_ids=context['removed_edge_ids'])
        over_edges = self.sample_edges_pi(nei_edges, hparams.max_attd_over)
        over_edges = self.add_selfloop_edges_pi(over_edges)
        return over_edges

    def get_prop_over_edges(self, hparams, context, attd_over_edges, attd_to_nodes):
        pass


class Reasoner(object):
    def __init__(self, hparams, context, graph):
        self.hparams = hparams
        self.context = context
        self.graph = graph

        graph_accessor = GraphAccessor(graph, extra_etypes='selfloop', activated_edges=context['activated_edges'])
        nodes_embs = NodeParams(graph_accessor.n_nodes, hparams.n_dims, context['device'])
        etype_embs = ETypeParams(graph_accessor.n_etypes, hparams.n_dims, context['device'])
        attention = Attention(graph_accessor.n_nodes)
        nodes_states = NodeStates(graph_accessor.n_nodes)

        self.engine = gre.Engine(hparams, context, graph_accessor, attention,
                                 node_params=nodes_embs, etype_params=etype_embs,
                                 node_states=nodes_states)

    def initialize(self, heads, tails, rels, di):
        """ [Param] heads (1D numpy): B
            [Param] tails (1D numpy): B
            [Param] rels (1D numpy): B
            [Param] di (1D numpy): B
        """
        batch_size = len(heads)
        idx = np.arange(batch_size)
        if self.hparams.removed_edges == 'h_t':
            removed_edge_ids = self.graph.get_edge_ids(np.stack([idx, heads, tails], 1), has_idx=True, via='v1v2')
        elif self.hparams.removed_edges == 'ht_th':
            idx = np.repeat(idx, 2)
            v1 = np.stack([heads, tails], 1).flatten()
            v2 = np.stack([tails, heads], 1).flatten()
            removed_edge_ids = self.graph.get_edge_ids(np.stack([idx, v1, v2], 1), has_idx=True, via='v1v2')
        elif self.hparams.removed_edges == 'h_t_r_di':
            removed_edge_ids = self.graph.get_edge_ids(np.stack([idx, heads, tails, rels, di], 1), has_idx=True, via='edge')
        elif self.hparams.removed_edges == 'ht_th_r':
            idx = np.repeat(idx, 4)
            v1 = np.stack([heads, heads, tails, tails], 1).flatten()
            v2 = np.stack([tails, tails, heads, heads], 1).flatten()
            r = np.repeat(rels, 4)
            pos_di = np.full_like(heads, const.POS_DI)
            neg_di = np.full_like(heads, const.NEG_DI)
            di = np.stack([pos_di, neg_di, pos_di, neg_di], 1).flatten()
            removed_edge_ids = self.graph.get_edge_ids(np.stack([idx, v1, v2, r, di], 1), has_idx=True, via='edge')
        else:
            raise ValueError
        self.context['removed_edge_ids'] = removed_edge_ids

        self.engine.initialize_attention(heads)
        self.engine.initialize_short_memory(heads, rels)

    def forward(self):
        ''' [Param] heads (tensor): B

            [Param] rels (tensor): B
            [Param] di (tensor): B
        '''
        for step in range(1, self.hparams.n_steps + 1):
            self.engine()


    def loss(self):
        pass
