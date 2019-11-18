""" Graph Accessor
"""
import numpy as np

import gre.fast_numpy as fnp
import gre.consts as const


class GraphAccessor(object):
    def __init__(self, graph, extra_nodes=None, extra_etypes='selfloop', activated_edges=None, deactivated_edges=None):
        """ extra_nodes: str or list or tuple
            extra_etypes: str or list or tuple
            activated_edges: (2D numpy) N x 4 where row = (v1, v2, et, di)
            deactivated_edges: (2D numpy) N x 4 where row = (v1, v2, et, di)
        """
        self.graph = graph
        self.v_str2id = graph.v_str2id.copy()
        self.v_id2str = graph.v_id2str.copy()
        self.et_str2id = graph.et_str2id.copy()
        self.et_id2str = graph.et_id2str.copy()

        self._add_extra_nodes(extra_nodes)
        self._add_extra_etypes(extra_etypes)
        self.n_nodes = len(self.v_str2id)
        self.n_etypes = len(self.et_str2id)

        self.edge_table = graph.edge_table  # (2D numpy): (e, v1, v2, et, di) sorted by (v1, v2, et, di)
        self.acti_edge_ids = set()
        self.acti_edge_stack = []

        self._activate_edges(activated_edges)
        self._deactivate_edges(deactivated_edges)

        self.edge_id_to_extra_et = dict()
        self.extra_et_to_edge_id = dict()

    def _add_extra_nodes(self, nodes_str):
        if nodes_str is None:
            return
        elif not isinstance(nodes_str, (list, tuple)):
            nodes_str = [nodes_str]

        for v in nodes_str:
            if v not in self.v_str2id:
                self.v_str2id[v] = len(self.v_str2id)
                self.v_id2str[self.v_str2id[v]] = v

    def _add_extra_etypes(self, etypes_str):
        if etypes_str is None:
            return
        elif not isinstance(etypes_str, (list, tuple)):
            etypes_str = [etypes_str]

        for et in etypes_str:
            if et not in self.et_str2id:
                self.et_str2id[et] = len(self.et_str2id)
                self.et_id2str[self.et_str2id[et]] = et

    def _get_edge_id_for_extra_etype(self, et_str):
        et = self.et_str2id[et_str]
        if et not in self.extra_et_to_edge_id:
            e_id = -(len(self.extra_et_to_edge_id) + 1)
            self.extra_et_to_edge_id[et] = e_id
            self.edge_id_to_extra_et[e_id] = et
        return self.extra_et_to_edge_id[et]

    def activate_edges(self, edge_ids=None, reset=False):
        """ [Param] edge_ids (None or list or tuple or set)
        """
        if edge_ids is None:
            self.acti_edge_ids = set(range(self.graph.n_edges))
        else:
            if reset:
                self.acti_edge_ids = set(edge_ids)
            else:
                self.acti_edge_ids |= set(edge_ids)

    def _activate_edges(self, activated_edges):
        edge_ids = None if activated_edges is None else self.graph.get_edge_ids(edges=activated_edges)
        self.activate_edges(edge_ids=edge_ids, reset=True)

    def deactivate_edges(self, edge_ids=None):
        """ [Param] edge_ids (None or list or tuple or set)
        """
        if edge_ids is not None:
            self.acti_edge_ids -= set(edge_ids)

    def _deactivate_edges(self, deactivated_edges):
        edge_ids = None if deactivated_edges is None else self.graph.get_edge_ids(edges=deactivated_edges)
        self.deactivate_edges(edge_ids=edge_ids)

    def stack_push_edges(self):
        self.acti_edge_stack.append(self.acti_edge_ids.copy())

    def stack_pop_edges(self):
        self.acti_edge_ids = self.acti_edge_stack.pop()

    def stack_peek_edges(self):
        self.acti_edge_ids = self.acti_edge_stack[-1]

    @property
    def edges(self):
        ids = sorted(list(self.acti_edge_ids))
        return self.edge_table[ids]

    @staticmethod
    def get_topk_nodes_pi(attn, k, threshold=0.0):
        """ [Param] attn (sparse tensor): (idx, v) -> 1 where (idx, v) takes 2 x N
            [Return] topk_nodes (2D numpy): (idx, v) sorted by (idx, v)
            [Note] 'pi' stands for 'per idx'
        """
        nodes = fnp.to_numpy(attn.indices()).T  # N x 2 where row = (idx, v)
        key_idx = nodes[:, 0]
        val_attn = fnp.to_numpy(attn.values())  # N
        if threshold > 0.0:
            mask = (val_attn >= threshold)
            nodes = nodes[mask]
            key_idx = nodes[:, 0]
            val_attn = val_attn[mask]
        ind = fnp.sorted_group_topk(key_idx, val_attn, k, largest=True)
        return nodes[ind]

    @staticmethod
    def get_largerequal_nodes_pi(attn, threshold):
        """ [Param] attn (sparse tensor): (idx, v) -> 1 where (idx, v) takes 2 x N
            [Return] le_nodes (2D numpy): (idx, v) sorted by (idx, v)
        """
        nodes = fnp.to_numpy(attn.indices()).T  # N x 2 where row = (idx, v)
        mask = (fnp.to_numpy(attn.values()) >= threshold)  # N
        return nodes[mask]

    def get_out_edges_pi(self, nodes, edges, selfloop_on_leaf=False, removed_edge_ids=None):
        """ [Param] nodes (2D numpy): (idx, v1) sorted by (idx, v1)
            [Param] edges (idx-invariant) (2D numpy): (e, v1, v2) sorted by (v1, v2)
            [Param] removed_edges_ids (idx-dependent) (2D numpy): (idx, e) sorted by (idx, e)
            [Return] out_edges (2D numpy): (idx, e, v1, v2) sorted by (idx, v1, v2)
        """
        if selfloop_on_leaf:
            assert 'selfloop' in self.et_str2id
            ind1, ind2 = fnp.sorted_left_group_join(nodes, edges, 0, 1, 1, mode='left')  # n_out_edges = len(ind1) = len(ind2)
            idx, v1 = nodes[ind1].T
            selfloop_e = self._get_edge_id_for_extra_etype('selfloop')
            v1_selfloop = np.stack([np.full_like(v1, selfloop_e), v1, v1], 1)  # n_out_edges x 3 where row = (-1, v1, v1)
            selfloop_cond = np.expand_dims(ind2 == -1, 1)
            out_edges = np.where(selfloop_cond, v1_selfloop, edges[ind2])  # n_out_edges x 3
            out_edges = np.concatenate([np.expand_dims(idx, 1), out_edges], 1)  # n_out_edges x 4 where row = (idx, e, v1, v2)
        else:
            ind1, ind2 = fnp.sorted_left_group_join(nodes, edges, 0, 1, 1, mode='inner')
            out_edges = np.concatenate([nodes[ind1][:, :1], edges[ind2]], 1)  # n_out_edges x 4 where row = (idx, e, v1, v2)

        if removed_edge_ids is not None:
            out_edges = self.filter_edges_pi(out_edges, removed_edge_ids)
        return out_edges

    @staticmethod
    def filter_edges_pi(edges, removed_edge_ids):
        """ [Param] edges (idx-dependent) (2D numpy): (idx, e, v1, v2) sorted by (idx, v1, v2)
            [Param] removed_edge_ids: (idx-dependent) (2D numpy): (idx, e) sorted by (idx, e)
            [Return] filtered_edges (2D numpy): (idx, e, v1, v2) sorted by (idx, v1, v2)
        """
        _, ind = fnp.fast_sorted_overlap(edges[:, :2], removed_edge_ids, return_arr1_indices=True, return_arr2_indices=False)
        return edges[ind]

    @staticmethod
    def sample_edges_pi(edges, k, per_v1=True):
        """ [Param] edges (idx-dependent) (2D numpy): (idx, e, v1, v2) sorted by (idx, v1, v2)
            [Return] sampled_edges (2D numpy): (idx, e, v1, v2) sorted by (idx, v1, v2)
        """
        r = np.random.rand(len(edges), 1)
        if per_v1:
            ind = fnp.sorted_group_topk(edges[:, [0, 2]], r, k)
        else:
            ind = fnp.sorted_group_topk(edges[:, 0], r, k)
        return edges[ind]

    def add_selfloop_edges_pi(self, edges):
        """ [Param] edges (idx-dependent) (2D numpy): (idx, e, v1, v2) sorted by (idx, v1, v2)
            [Return] aug_edges (2D numpy): (idx, e, v1, v2) sorted by (idx, v1, v2) with selfloop (idx, -1, v1, v1)
            [Note] when using add_selfloop_edges_pi(), do not set selfloop_on_leaf=True in get_out_edges_pi()
        """
        ind = fnp.sorted_unique(edges[:, [0, 2]])
        idx, v1 = edges[ind].T
        selfloop_e = self._get_edge_id_for_extra_etype('selfloop')
        v1_selfloop = np.stack([idx, np.full_like(v1, selfloop_e), v1, v1], 1)  # (idx, -1, v1, v1)
        aug_edges = fnp.sort(np.concatenate([edges, v1_selfloop]), dim=[0, 2, 3])
        return aug_edges

    def get_etypes(self, edge_ids):
        """ [Param] edge_ids (1D numpy): (e) including e < 0 for extra etypes
            [Return] etypes (2D numpy): (et, di)
        """
        etypes = self.edge_table[edge_ids][:, -2:]  # (et, di)
        for e, et in self.edge_id_to_extra_et.items():
            ext_etypes = np.full_like(etypes, et)
            ext_etypes[:, 1] = const.POS_DI
            etypes = np.where(edge_ids == e, ext_etypes, etypes)
        return etypes

    def get_attd_from_nodes(self, hparams, context, attn):
        """ [Param] attn (sparse tensor): (idx, v) -> 1
            [Return] attd_from_nodes (2D numpy): (idx, v1) sorted by (idx, v1)
        """
        raise NotImplementedError

    def get_attd_to_nodes(self, hparams, context, attn):
        """ [Param] attn (sparse tensor): (idx, v) -> 1
            [Return] attd_to_nodes (2D numpy): (idx, v2) sorted by (idx, v2)
        """
        raise NotImplementedError

    def get_attd_over_edges(self, hparams, context, attd_from_nodes):
        """ [Param] attd_from_nodes (2D numpy): (idx, v1) sorted by (idx, v1)
            [Return] attd_over_edges (2D numpy): (idx, e, v1, v2) sorted by (idx, v1, v2)
        """
        raise NotImplementedError

    def get_prop_over_edges(self, hparams, context, attd_over_edges, attd_to_nodes):
        """ [Param] attd_over_edges (2D numpy): (idx, e, v1, v2) sorted by (idx, e, v1, v2)
            [Param] attd_to_nodes (2D numpy): (idx, v2) sorted by (idx, v2)
            [Return] prop_over_edges (2D numpy): (idx, e, v1, v2) sorted by (idx, e, v1, v2)
        """
        raise NotImplementedError
