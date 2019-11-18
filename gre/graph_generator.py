from collections import defaultdict
import numpy as np

import gre.fast_numpy as fnp
import gre.consts as const

class Graph(object):
    def __init__(self):
        self.v_str2id = dict()
        self.et_str2id = dict()
        self.edge_table = np.array([])

    @property
    def n_nodes(self):
        return len(self.v_str2id)

    @property
    def n_etypes(self):
        return len(self.et_str2id)

    @property
    def n_edges(self):
        return len(self.edge_table)

    def get_edges(self, edge_ids=None, has_idx=False, return_etypes=False):
        """ [Param] edge_ids (1D numpy) or (2D numpy) row = (idx, e)
            [Return] edges (2D numpy) row = (e, v1, v2) or (e, v1, v2, et, di) or row = (idx, e, v1, v2) or (idx, e, v1, v2, et, di)
        """
        raise NotImplementedError

    def get_edge_ids(self, edges=None, has_idx=False, via='edge'):
        """ [Param] edges (1D numpy) or (2D numpy)
                1) via = 'edge': row = (v1, v2, et, di) or (idx, v1, v2, et, di)
                2) via = 'v1': row = (v1,) or (idx, v1)
                3) via = 'v2': row = (v2,) or (idx, v2)
                4) via = 'v1v2': row = (v1, v2) or (idx, v1, v2)
            [Param] via ('edge', 'v1', 'v2', 'v1v2')
            [Return] edge_ids (1D numpy or list(1D numpy))
        """
        raise NotImplementedError


class LoadedGraph(Graph):
    """ Data file format: <v1>\t<edge_type>\t<v2>
        Edge format: (v1, v2, edge_type, direction) where edge_type in {0, 1, 2, ...} and direction in {-1, 1}

        edge_table (2D numpy): (e, v1, v2, et, di) sorted by (v1, v2, et, di), consecutively increasing e starting with 0
        index (dict): (v1, v2, et, di) -> e
        index_on_v1 (dict): v1 -> set(e)
        index_on_v2 (dict): v2 -> set(e)
        index_on_v1v2 (dict): (v1, v2) -> set(e)
    """
    def __init__(self, file_path, add_reverse=False, identify_reverse_fn=None):
        super(LoadedGraph, self).__init__()
        edge_set = self._load_edges(file_path, identify_reverse_fn)
        if add_reverse:
            edge_set = self._add_reverse_edges(edge_set)

        self.v_str2id, self.v_id2str, self.et_str2id, self.et_id2str = self._make_dict(edge_set)

        edges = self._convert_to_numpy(edge_set, self.v_str2id, self.et_str2id)
        self.edge_table = self._create_edge_table(edges)
        self.index, self.index_on_v1, self.index_on_v2, self.index_on_v1v2 = self._create_edge_index(self.edge_table)

    @staticmethod
    def _load_edges(path, identify_reverse_fn):
        edge_set = set()
        with open(path) as fin:
            for line in fin:
                v1, etype, v2 = line.strip().split('\t')
                if identify_reverse_fn is None:
                    edge_set.add((v1, v2, etype, const.POS_DI))
                else:
                    is_reverse, orig_etype = identify_reverse_fn(etype)
                    edge_set.add((v1, v2, orig_etype, const.NEG_DI) if is_reverse else (v1, v2, etype, const.POS_DI))
        return edge_set

    @staticmethod
    def _add_reverse_edges(edge_set):
        reverse_edges = set()
        for v1, v2, etype, di in edge_set:
            if di == 1 and v1 != v2:
                reverse_edges.add((v2, v1, etype, const.NEG_DI))
        edge_set.update(reverse_edges)
        return edge_set

    @staticmethod
    def _make_dict(edge_set):
        v_str2id, v_id2str, et_str2id, et_id2str = dict(), dict(), dict(), dict()
        for v1, v2, etype, di in edge_set:
            v_str2id.setdefault(v1, len(v_str2id))
            v_id2str[v_str2id[v1]] = v1
            v_str2id.setdefault(v2, len(v_str2id))
            v_id2str[v_str2id[v2]] = v2
            et_str2id.setdefault(etype, len(et_str2id))
            et_id2str[et_str2id[etype]] = etype
        return v_str2id, v_id2str, et_str2id, et_id2str

    @staticmethod
    def _convert_to_numpy(edge_set, v_str2id, et_str2id):
        return np.array([[v_str2id[v1], v_str2id[v2], et_str2id[etype], di]
                         for v1, v2, etype, di in edge_set])

    @staticmethod
    def _create_edge_table(edges):
        ind = fnp.argsort(edges, dim=[0, 1, 2, 3])
        ids = np.expand_dims(np.arange(len(edges)), 1)
        table = np.concatenate([ids, edges[ind]], 1)
        return table

    def _create_edge_index(self, table):
        index, index_on_v1, index_on_v2, index_on_v1v2 = \
            dict(), defaultdict(set), defaultdict(set), defaultdict(set)
        index_str, index_on_v1_str, index_on_v2_str, index_on_v1v2_str = \
            dict(), defaultdict(set), defaultdict(set), defaultdict(set)
        for row in table:
            e, v1, v2, et, di = row
            index[(v1, v2, et, di)] = e
            index_on_v1[v1].add(e)
            index_on_v2[v2].add(e)
            index_on_v1v2[(v1, v2)].add(e)

            index_str[(self.v_id2str[v1], self.v_id2str[v2], self.et_id2str[et], di)] = e
            index_on_v1_str[self.v_id2str[v1]].add(e)
            index_on_v2_str[self.v_id2str[v2]].add(e)
            index_on_v1v2_str[(self.v_id2str[v1], self.v_id2str[v2])].add(e)
        return index, index_on_v1, index_on_v2, index_on_v1v2

    def get_edges(self, edge_ids=None, has_idx=False, return_etypes=False):
        """ [Param] edge_ids (1D numpy) sorted, or (2D numpy) row = (idx, e) sorted
            [Return] edges (2D numpy) row = (e, v1, v2) or (e, v1, v2, et, di) sorted,
                                   or row = (idx, e, v1, v2) or (idx, e, v1, v2, et, di) sorted
        """
        if edge_ids is None:
            if return_etypes:
                return self.edge_table
            else:
                return self.edge_table[:, :3]
        else:
            if has_idx:
                edge_ids = edge_ids[:, 1]
                idx = np.expand_dims(edge_ids[:, 0], 1)
                if return_etypes:
                    return np.concatenate([idx, self.edge_table[edge_ids]], 1)
                else:
                    return np.concatenate([idx, self.edge_table[edge_ids, :3]], 1)
            else:
                edge_ids = edge_ids.flatten()
                if return_etypes:
                    return self.edge_table[edge_ids]
                else:
                    return self.edge_table[edge_ids, :3]

    def get_edge_ids(self, edges=None, has_idx=False, via='edge'):
        """ [Param] edges (1D numpy) or (2D numpy)
                1) via = 'edge': row = (v1, v2, et, di) or (idx, v1, v2, et, di)
                2) via = 'v1': row = (v1,) or (idx, v1)
                3) via = 'v2': row = (v2,) or (idx, v2)
                4) via = 'v1v2': row = (v1, v2) or (idx, v1, v2)
            [Param] via ('edge', 'v1', 'v2', 'v1v2')
            [Return] edge_ids (1D numpy) sorted for has_idx == False, or (2D numpy) sorted for has_idx == True
        """
        if edges is None:
            return self.edge_table[:, 0]
        else:
            edges = np.expand_dims(edges, 1) if edges.ndim == 1 else edges
            if via == 'edge':
                if has_idx:
                    return fnp.sort(np.array([[edge[0], self.index[tuple(edge[1:])]]
                                              for edge in edges if tuple(edge[1:]) in self.index]))
                else:
                    return fnp.sort(np.array([self.index[tuple(edge)]
                                              for edge in edges if tuple(edge) in self.index]))
            elif via == 'v1':
                if has_idx:
                    edge_ids = [(v1[0], np.array(list(self.index_on_v1[v1[1]])))
                                for v1 in edges if v1[1] in self.index_on_v1]
                    edge_ids = [np.stack([np.repeat(idx, len(e_ids)), e_ids], 1) for idx, e_ids in edge_ids]
                    return fnp.sort(np.concatenate(edge_ids, 0))
                else:
                    edge_ids = [np.array(list(self.index_on_v1[v1]))
                                for v1 in edges if v1 in self.index_on_v1]
                    return fnp.sort(np.concatenate(edge_ids, 0))
            elif via == 'v2':
                if has_idx:
                    edge_ids = [(v2[0], np.array(list(self.index_on_v2[v2[1]])))
                                for v2 in edges if v2[1] in self.index_on_v2]
                    edge_ids = [np.stack([np.repeat(idx, len(e_ids)), e_ids], 1) for idx, e_ids in edge_ids]
                    return fnp.sort(np.concatenate(edge_ids, 0))
                else:
                    edge_ids = [np.array(list(self.index_on_v2[v2]))
                                for v2 in edges if v2 in self.index_on_v2]
                    return fnp.sort(np.concatenate(edge_ids, 0))
            elif via == 'v1v2':
                if has_idx:
                    edge_ids = [(v1v2[0], np.array(list(self.index_on_v1v2[v1v2[1:]])))
                                for v1v2 in edges if v1v2[1:] in self.index_on_v1v2]
                    edge_ids = [np.stack([np.repeat(idx, len(e_ids)), e_ids], 1) for idx, e_ids in edge_ids]
                    return fnp.sort(np.concatenate(edge_ids, 0))
                else:
                    edge_ids = [np.array(list(self.index_on_v1v2[v1v2]))
                                for v1v2 in edges if v1v2 in self.index_on_v1v2]
                    return fnp.sort(np.concatenate(edge_ids, 0))
            else:
                raise ValueError


class GeneratedGraph(object):
    pass


class PredefinedGraph(object):
    pass
