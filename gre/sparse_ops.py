import torch

import gre.fast_numpy as fnp


def to_indices(indices, device=None):
    """ [Param] indices (2D numpy): e.g. B x 2 where row = (i0, i1)
        [Return] indices (tensor): e.g. 2 x B
    """
    return torch.from_numpy(indices.T).to(device=device)


def idx_indices(indices):
    """ [Param] indices (tensor): B or k x B where col = (i1, ..., ik) sorted and unique
        [Return] out (tensor): 2 x B or (k+1) x B where col = (idx, i1, ..., ik)
    """
    assert indices.dim() in (1, 2)
    if indices.dim() == 1:
        return torch.stack([torch.arange(indices.size(0)).to(indices), indices], 0)
    else:
        return torch.cat([torch.arange(indices.size(1)).unsqueeze(0).to(indices), indices], 0)


def sparse_one_hot(indices, sparse_size=None):
    """ [Param] indices (tensor): B or k x B where col = (i1, ..., ik) sorted and unique
        [Return] out (sparse tensor): (i1, ..., ik) -> 1 where (i1, ..., ik) takes k x B
    """
    if indices.dim() == 1:
        indices = torch.unsqueeze(indices, 0)  # 1 x B
    assert indices.dim() == 2
    values = torch.ones(indices.size(1)).to(device=indices.device)
    if sparse_size is None:
        return torch.sparse_coo_tensor(indices, values).coalesce()
    else:
        return torch.sparse_coo_tensor(indices, values, size=sparse_size).coalesce()


def sparse_fill(indices, val, sparse_size=None):
    """ [Param] indices (tensor): B or k x B where col = (i1, ..., ik) sorted and unique
        [Param] val (int or float or tensor)
        [Return] out (sparse tensor): (i1, ..., ik) -> ... where (i1, ..., ik) takes k x B
    """
    if indices.dim() == 1:
        indices = torch.unsqueeze(indices, 0)  # 1 x B
    assert indices.dim() == 2
    val = torch.tensor(val).to(device=indices.device) if isinstance(val, (int, float)) else val
    values = torch.unsqueeze(val, 0).repeat([indices.size(1)] + [1] * val.dim())
    if sparse_size is None:
        return torch.sparse_coo_tensor(indices, values).coalesce()
    else:
        return torch.sparse_coo_tensor(indices, values, size=sparse_size).coalesce()


def index_select_to_sparse(dense, dim_d, indices, dim_s, sparse_size=None, fn_values=None):
    """ [Param] dense (tensor or list of tensors): e.g. D1 x D2
        [Param] dim_d (int or list of int): e.g. 0
        [Param] indices (tensor): n_sparse_dims x nnz, e.g. col: (i0, i1) sorted and unique
        [Param] dim_s (int or list of int): e.g. 1
        [Return] out (sparse tensor): e.g. (i0, i1) -> D2 where (i0, i1) takes n_sparse_dims x nnz
    """
    if isinstance(dense, (list, tuple)):
        values = []
        for i, den in enumerate(dense):
            d_d = dim_d[i] if isinstance(dim_d, (list, tuple)) else dim_d
            d_s = dim_s[i] if isinstance(dim_s, (list, tuple)) else dim_s
            values.append(torch.index_select(den, d_d, indices[d_s]))
        values = torch.cat(values, -1)  # ... x (n_dims*l)
    else:
        values = torch.index_select(dense, dim_d, indices[dim_s])  # ... x n_dims

    if fn_values and callable(fn_values):
        values = fn_values(values)
    if sparse_size is None:
        return torch.sparse_coo_tensor(indices, values).coalesce()
    else:
        return torch.sparse_coo_tensor(indices, values, size=list(sparse_size) + list(values.size()[1:])).coalesce()


def index_to_sparse(dense, indices, dim_s, sparse_size=None, fn_values=None):
    """ [Param] dense (tensor): e.g. D1 x D2 x D3
        [Param] indices (tensor): n_sparse_dims x nnz, e.g. col: (i0, i1, i2) sorted and unique
        [Param] dim_s (int or list or tuple): e.g. (1, 2)
        [Return] out (sparse tensor): e.g. (i0, i1, i2) -> D3 where (i0, i1, i2) takes n_sparse_dims x nnz
    """
    values = dense[indices[dim_s]] if isinstance(dim_s, int) else dense[indices[list(dim_s)].unbind()]
    if fn_values and callable(fn_values):
        values = fn_values(values)
    if sparse_size is None:
        return torch.sparse_coo_tensor(indices, values).coalesce()
    else:
        return torch.sparse_coo_tensor(indices, values, size=list(sparse_size) + list(values.size()[1:])).coalesce()


def none_to_sparse(sparse_size, dense_size, device=None):
    """ [Param] sparse_size (list or tuple)
        [Param] dense_size (list or tuple)
        [Return] out (empty sparse tensor)
    """
    values = torch.empty([0] + list(dense_size)).to(device=device)
    indices = torch.empty([len(sparse_size), 0], dtype=torch.int64).to(device=device)
    return torch.sparse_coo_tensor(indices, values, size=list(sparse_size) + list(dense_size)).coalesce()


def indices_to_sparse(indices, val, sparse_size=None, device=None):
    """ [Param] indices (tensor): n_sparse_dims x N
        [Param] val (int or float)
        [Return] out (sparse tensor)
    """
    values = torch.tensor([val]).repeat(indices.size(1)).to(device)
    if sparse_size is None:
        return torch.sparse_coo_tensor(indices, values).coalesce()
    else:
        return torch.sparse_coo_tensor(indices, values, size=list(sparse_size) + [1]).coalesce()


def sparse_index(src, indices, sparse_size=None):
    """ [Param] src (sparse tensor): e.g. (i1, i2) -> D where (i1, i2): 2 x N
        [Param] indices (tensor): n_sparse_dims x M, e.g. (j1, j2): 2 x M sorted and unique
        [Return] out (sparse tensor): e.g. (j1, j2) -> D where (j1, j2): 2 x M
    """
    src_indices = src.indices()
    assert src_indices.size(0) == indices.size(0)
    src_values = src.values()
    values = torch.zeros([indices.size(1)] + list(src_values.size()[1:])).to(src_values)
    ind1, _, ind2, _ = fnp.fast_sorted_overlap(fnp.to_numpy(indices).T, fnp.to_numpy(src_indices).T)
    values[ind1] = src_values[ind2]
    if sparse_size is None:
        return torch.sparse_coo_tensor(indices, values, size=src.size()).coalesce()
    else:
        return torch.sparse_coo_tensor(indices, values, size=list(sparse_size) + list(values.size()[1:])).coalesce()


def sparse_broadcast(src, indices, dim=None, n_dims=None, sparse_size=None):
    """ [Param] src (sparse tensor): e.g. (i1, i2) -> D where (i1, i2): 2 x N
        [Param] indices (tensor): n_sparse_dims x M, e.g. (j0, j1, j2): 3 x M sorted and unique
        [Param] dim (int or list or tuple): e.g. (1, 2)
        [Param] n_dims (int)
        [Return] out (sparse tensor): e.g. (j0, j1, j2) -> D where (j1, j2, j3): 3 x M
    """
    assert dim is not None or n_dims is not None
    if dim is not None:
        indices2 = indices[dim] if isinstance(dim, int) else indices[list(dim)]
        indices2 = fnp.to_numpy(indices2).T
        sort_ind = fnp.argsort(indices2)
        indices2 = indices2[sort_ind]
        src_indices = src.indices()
        assert src_indices.size(0) == indices2.shape[1]
        src_values = src.values()
        values = torch.zeros([indices.size(1)] + list(src_values.size()[1:])).to(src_values)
        ind1, ind2 = fnp.sorted_matched_pairs(indices2, fnp.to_numpy(src_indices).T)
        values[ind1] = src_values[ind2]
        sortback_ind = fnp.argsort(sort_ind)
        values = values[sortback_ind]
        if sparse_size is None:
            return torch.sparse_coo_tensor(indices, values).coalesce()
        else:
            return torch.sparse_coo_tensor(indices, values, size=list(sparse_size) + list(values.size()[1:])).coalesce()
    else:
        return fast_sparse_broadcast(src, indices, n_dims, sparse_size=sparse_size)


def fast_sparse_broadcast(src, indices, n_dims, sparse_size=None):
    """ [Param] src (sparse tensor): e.g. (i1, i2) -> D where (i1, i2): 2 x N
        [Param] indices (tensor): n_sparse_dims x M, e.g. (j1, j2, j3): 3 x M sorted and unique, but (j1, j2) sorted but not unique
        [Param] n_dims (int): e.g. 2
        [Return] out (sparse tensor): e.g. (j1, j2, j3) -> D where (j1, j2, j3): 3 x M
    """
    indices2 = indices[:n_dims]
    src_indices = src.indices()
    assert src_indices.size(0) == indices2.size(0)
    src_values = src.values()
    values = torch.zeros([indices.size(1)] + list(src_values.size()[1:])).to(src_values)
    ind1, ind2 = fnp.sorted_matched_pairs(fnp.to_numpy(indices2).T, fnp.to_numpy(src_indices).T)
    values[ind1] = src_values[ind2]
    if sparse_size is None:
        return torch.sparse_coo_tensor(indices, values).coalesce()
    else:
        return torch.sparse_coo_tensor(indices, values, size=list(sparse_size) + list(values.size()[1:])).coalesce()


def sparse_sum(src, dim, indices=None):
    """ [Param] src (sparse tensor): e.g. (i0, i1, i2) -> D where (i0, i1, i2): 3 x N
        [Param] dim (int or list or tuple): e.g. 0
        [Param] indices (tensor): n_sparse_dims x M, e.g. (j1, j2): 2 x M
        [Return] out (sparse tensor): e.g. (j1, j2) -> D where (j1, j2): 2 x M
    """
    src2 = torch.sparse.sum(src, dim)
    out = src2 if indices is None else sparse_index(src2, indices)
    return out


def sparse_count(src, dim, indices=None):
    """ [Param] src (sparse tensor): e.g. (i0, i1, i2) -> D where (i0, i1, i2): 3 x N
        [Param] dim (int or list or tuple): e.g. 0
        [Param] indices (tensor): n_sparse_dims x M, e.g. (j1, j2): 2 x M
        [Return] out (sparse tensor): e.g. (j1, j2) ->
    """
    src_indices = src.indices()
    dim = [dim] if isinstance(dim, int) else dim
    dim2 = [d for d in range(src_indices.size(0)) if d not in dim]
    indices2, values2 = torch.unique(src_indices[dim2], return_counts=True, dim=1)
    out = torch.sparse_coo_tensor(indices2, values2, size=[src.size(d) for d in dim2]).coalesce()
    return out if indices is None else sparse_index(out, indices)


def sparse_reduce_broacast(src, dim_s, indices, dim_i=None, n_dims=None, sparse_size=None, mode='sum'):
    """ [Param] src (sparse tensor): e.g. (i0, i1, i2) -> D where (i0, i1, i2): 3 x N
        [Param] dim_s (int or list or tuple) e.g. 0
        [Param] indices (tensor): n_sparse_dims x M, e.g. (j1, j2, j3): 3 x M
        [Param] dim_i (int or list or tuple) e.g. (0, 1)
        [Param] n_dims (int)
        [Param] mode ('sum', 'count')
        [Return] out (sparse tensor): e.g. (j1, j2, j3) -> D
    """
    assert mode in ('sum', 'count')
    if dim_i is None:
        indices2 = indices[:n_dims]
    else:
        indices2 = indices[dim_i] if isinstance(dim_i, int) else indices[list(dim_i)]
    out = sparse_sum(src, dim_s, indices2) if mode == 'sum' else sparse_count(src, dim_s, indices2)
    sparse_broadcast(out, indices, dim=dim_i, n_dims=n_dims, sparse_size=sparse_size)


def sparse_filter(src, idx_mask):
    """ [Param] src (sparse tensor): e.g. (i0, i1, i2) -> D where (i0, i1, i2): 3 x N
        [Param] sparse_mask (tensor): e.g. N
        [Return] out (sparse tensor): e.g. (i0, i1, i2) -> D where (i0, i1, i2): 3 x M
    """
    src_indices = src.indices()
    src_values = src.values()
    assert src_indices.size(1) == idx_mask.size(0)
    return torch.sparse_coo_tensor(src_indices[:, idx_mask], src_values[idx_mask], size=src.size()).coalesce()


def sparse_cat(src_list, dim):
    """ [Param] src_list: each sparse tensor in src_list should share the same indices
        [Param] dim: for the dense part
    """
    for src in src_list:
        assert src.indices.size() == src_list[0].indices.size()
    indices = src_list[0].indices
    values = torch.cat([src.values() for src in src_list], dim)
    size = list(src_list[0].size())
    dense_size = list(values.size()[1:])
    size[-len(dense_size):] = dense_size
    return torch.sparse_coo_tensor(indices, values, size=size).coalesce()
