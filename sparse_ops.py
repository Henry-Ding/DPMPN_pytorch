


def broadcast_sparse(sp_tensor, indices, dim):
    ''' [Param] sp_tensor (sparse tensor): e.g. (i, j) -> ...
        [Param] indices (2D numpy): e.g. (i, j, k)
        [Param] dim (int or tuple or list): e.g. [0, 1]
        [Return] bc_sp_tensor (sparse tensor): e.g. (i, j, k) -> ...
        [Note] requires sp_tensor.indices == deduplicate(indices[:, dim])
    '''
    pass
