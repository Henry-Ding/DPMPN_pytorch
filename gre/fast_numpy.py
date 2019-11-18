from numba import jit
import numpy as np


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def combo_view(arr):
    """ [Param] arr (2D numpy) or (1D numpy)
        [Return] combo_arr (1D numpy)
    """
    c_order_arr = arr if arr.flags['C_CONTIGUOUS'] and arr.flags['OWNDATA'] else arr.copy()
    if c_order_arr.ndim == 1:
        return c_order_arr
    elif c_order_arr.ndim == 2:
        if c_order_arr.shape[1] == 1:
            return c_order_arr.flatten()
        else:
            return c_order_arr.view([('', c_order_arr.dtype)] * c_order_arr.shape[1]).flatten()
    else:
        raise ValueError('`arr` must be a 1D or 2D numpy array')


def argsort(arr, dim=None):
    """ [Param] arr (2D numpy) where row = (i1, i2, ..., ik), or (1D numpy)
        [Return] indices (1D numpy)
    """
    selected_arr = arr if dim is None else arr[:, dim]
    combo_arr = combo_view(selected_arr)
    indices = np.argsort(combo_arr)
    return indices


def topk_smallest(arr, k):
    """ [Param] arr (1D numpy)
        [Return] indices (1D numpy) sorted, with length of min(k, len(arr))
    """
    return np.sort(np.argpartition(arr, min(k - 1, len(arr) - 1))[:min(k, len(arr))])


def sorted_group_topk(keys, values, k, largest=True):
    """ [Param] keys (2D numpy) where row = (i1, i2, ..., ik) sorted by all, or (1D numpy) sorted
        [Param] values (2D numpy) where row = (v1, v2, ..., vl), or (1D numpy)
        [Return] indices (1D numpy) sorted
    """
    combo_keys = combo_view(keys)
    combo_values = combo_view(-values if largest else values)
    group_mask = (combo_keys[1:] != combo_keys[:-1])
    n_rows = len(combo_keys)
    group_partition = np.concatenate([np.array([0]), np.arange(1, n_rows)[group_mask], np.array([n_rows])])
    indices = np.concatenate([beg + topk_smallest(combo_values[beg:end], k)
                              for beg, end in zip(group_partition[:-1], group_partition[1:])])
    return indices


@jit(nopython=True)
def _compare(arr1, arr2):
    """ [Param] arr1 (1D numpy)
        [Param] arr2 (1D numpy)
        [Return] sign: {-1, 0, 1}
        [Note] arr1.shape == arr2.shape
    """
    for i in range(len(arr1)):
        sign = np.sign(arr1[i] - arr2[i])
        if sign == 0:
            continue
        else:
            return sign
    return 0


@jit(nopython=True)
def _equal(arr1, arr2):
    """ [Param] arr1 (1D numpy)
        [Param] arr2 (1D numpy)
        [Return] True or False
        [Note] arr1.shape == arr2.shape
    """
    for i in range(len(arr1)):
        if arr1[i] != arr2[i]:
            return False
    return True


@jit(nopython=True)
def _write(arr_list, arr, index, values):
    """ [Param] arr_list (list)
        [Param] arr (1D numpy)
        [Param] index (int)
        [Param] values (1D numpy)
    """
    num = len(values)
    if index + num > len(arr):
        arr_list.append(arr[:index].copy())
        index = 0
    arr[index:(index+num)] = values
    index += num
    return index


@jit(nopython=True)
def _sorted_join(arr1, arr2, mode):
    """ [Param] arr1 (2D numpy) sorted
        [Param] arr2 (2D numpy) sorted
        [Param] mode: {'inner', 'outer', 'left', 'right'}
        [Return] indices1_list
        [Return] indices2_list
    """
    indices1_list = [np.empty(0, dtype=np.int64)]
    indices2_list = [np.empty(0, dtype=np.int64)]
    len1, len2 = len(arr1), len(arr2)
    indices1 = np.empty(5, dtype=np.int64)
    indices2 = np.empty(5, dtype=np.int64)

    i1, i2 = 0, 0
    left_p, right_p = 0, 0
    while i1 < len1 and i2 < len2:
        cmp = _compare(arr1[i1], arr2[i2])
        if cmp == 0:
            j1 = i1 + 1
            while j1 < len1 and _equal(arr1[i1], arr1[j1]):
                j1 += 1
            j2 = i2 + 1
            while j2 < len2 and _equal(arr2[i2], arr2[j2]):
                j2 += 1
            for left_idx in range(i1, j1):
                right_idx = np.arange(i2, j2)
                left_p = _write(indices1_list, indices1, left_p, np.full(len(right_idx), left_idx))
                right_p = _write(indices2_list, indices2, right_p, right_idx)
            i1 = j1
            i2 = j2
        elif cmp == -1:
            if mode == 'left' or mode=='outer':
                left_p = _write(indices1_list, indices1, left_p, np.array([i1]))
                right_p = _write(indices2_list, indices2, right_p, np.array([-1]))
            i1 += 1
        else:
            if mode == 'right' or mode=='outer':
                left_p = _write(indices1_list, indices1, left_p, np.array([-1]))
                right_p = _write(indices2_list, indices2, right_p, np.array([i2]))
            i2 += 1

    while i1 < len1:
        if mode == 'left' or mode == 'outer':
            left_p = _write(indices1_list, indices1, left_p, np.array([i1]))
            right_p = _write(indices2_list, indices2, right_p, np.array([-1]))
        i1 += 1
    while i2 < len2:
        if mode == 'right' or mode == 'outer':
            left_p = _write(indices1_list, indices1, left_p, np.array([-1]))
            right_p = _write(indices2_list, indices2, right_p, np.array([i2]))
        i2 += 1

    indices1_list.append(indices1[:left_p])
    indices2_list.append(indices2[:right_p])
    return indices1_list[1:], indices2_list[1:]


def sorted_join(left_arr, right_arr, left_dim, right_dim, mode='inner'):
    """ [Param] left_arr (2D numpy): sorted on dimensions left_dim
        [Param] right_arr (2D numpy): sorted on dimensions right_dim
        [Param] left_dim: int or tuple or list
        [Param] right_dim: int or tuple or list
        [Param] mode: {'inner', 'outer', 'left', 'right'}
        [Return] indices1 (-1 means empty) sorted except -1
        [Return] indices2 (-1 means empty) sorted except -1
    """
    if isinstance(left_dim, int):
        left_dim = [left_dim]
    if isinstance(right_dim, int):
        right_dim = [right_dim]

    left_key = left_arr[:, left_dim]
    right_key = right_arr[:, right_dim]
    indices1_list, indices2_list = _sorted_join(left_key, right_key, mode)
    indices1 = np.concatenate(indices1_list)
    indices2 = np.concatenate(indices2_list)
    return indices1, indices2


@jit(nopython=True)
def _sorted_left_group_join(group_partition, arr1, arr2, mode):
    indices1_list = [np.empty(0, dtype=np.int64)]
    indices2_list = [np.empty(0, dtype=np.int64)]
    for beg, end in zip(group_partition[:-1], group_partition[1:]):
        ind1_list, ind2_list = _sorted_join(arr1[beg:end], arr2, mode)
        for ind1, ind2 in zip(ind1_list, ind2_list):
            ind1 += beg
            indices1_list.append(ind1)
            indices2_list.append(ind2)
    return indices1_list[1:], indices2_list[1:]


def sorted_left_group_join(left_arr, right_arr, left_group_dim, left_dim, right_dim, mode='inner'):
    """ [Param] left_arr (2D numpy): sorted on dimensions left_group_dim + left_dim
        [Param] right_arr (2D numpy): sorted on dimensions right_dim
        [Param] left_group_dim: int or tuple or list
        [Param] left_dim: int or tuple or list
        [Param] right_dim: int or tuple or list
        [Param] mode: {'inner', 'left'}
        [Return] indices1 (-1 means empty) sorted except -1
        [Return] indices2 (-1 means empty) sorted except -1
    """
    if isinstance(left_group_dim, int):
        left_group_dim = [left_group_dim]
    if isinstance(left_dim, int):
        left_dim = [left_dim]
    if isinstance(right_dim, int):
        right_dim = [right_dim]

    left_group = left_arr[:, left_group_dim]
    left_group = combo_view(left_group)
    group_mask = (left_group[1:] != left_group[:-1])
    n_rows = len(left_group)
    group_partition = np.concatenate([np.array([0]), np.arange(1, n_rows)[group_mask], np.array([n_rows])])
    indices1_list, indices2_list = \
        _sorted_left_group_join(group_partition, left_arr[:, left_dim], right_arr[:, right_dim], mode)
    indices1 = np.concatenate(indices1_list)
    indices2 = np.concatenate(indices2_list)
    return indices1, indices2


def intersect(arr1, arr2, assume_unique=False):
    """ [Param] arr1 (2D numpy) where row = (i1, i2, ..., ik), or (1D numpy)
        [Param] arr2 (2D numpy) where row = (i1, i2, ..., ik), or (1D numpy)
        [Return] indices1: only return the indices of the first occurrences of the common values in arr1
        [Return] indices2: only return the indices of the first occurrences of the common values in arr2
    """
    combo_arr1 = combo_view(arr1)
    combo_arr2 = combo_view(arr2)
    _, indices1, indices2 = np.intersect1d(combo_arr1, combo_arr2, assume_unique=assume_unique, return_indices=True)
    return indices1, indices2


def sorted_overlap(arr1, arr2, return_arr1_indices=True, return_arr2_indices=True):
    """ [Param] arr1 (2D numpy) where row = (i1, i2, ..., ik), or (1D numpy), sorted by all
        [Param] arr2 (2D numpy) where row = (i1, i2, ..., ik), or (1D numpy), sorted by all
        [Return] indices_1_in_2
        [Return] indices_1_notin_2
        [Return] indices_2_in_1
        [Return] indices_2_notin_1
    """
    combo_arr1 = combo_view(arr1)
    combo_arr2 = combo_view(arr2)

    results = []
    if return_arr1_indices:
        sort_left_2 = combo_arr2.searchsorted(combo_arr1, side='left')
        sort_right_2 = combo_arr2.searchsorted(combo_arr1, side='right')
        indices_1_in_2 = (sort_right_2 - sort_left_2 > 0).nonzero()[0]
        indices_1_notin_2 = (sort_right_2 - sort_left_2 == 0).nonzero()[0]
        results += [indices_1_in_2, indices_1_notin_2]

    if return_arr2_indices:
        sort_left_1 = combo_arr1.searchsorted(combo_arr2, side='left')
        sort_right_1 = combo_arr1.searchsorted(combo_arr2, side='right')
        indices_2_in_1 = (sort_right_1 - sort_left_1 > 0).nonzero()[0]
        indices_2_notin_1 = (sort_right_1 - sort_left_1 == 0).nonzero()[0]
        results += [indices_2_in_1, indices_2_notin_1]
    return results


@jit(nopython=True)
def _fast_sorted_overlap(arr1, arr2):
    """ [Param] arr1 (2D numpy)
        [Param] arr2 (2D numpy)
    """
    len1, len2 = len(arr1), len(arr2)
    mask1 = np.zeros(len1, dtype=np.bool8)
    mask2 = np.zeros(len2, dtype=np.bool8)
    i1, i2 = 0, 0
    while i1 < len1 and i2 < len2:
        while 0 < i1 < len1 and _equal(arr1[i1], arr1[i1 - 1]):
            mask1[i1] = mask1[i1 - 1]
            i1 += 1
        while 0 < i2 < len2 and _equal(arr2[i2], arr2[i2 - 1]):
            mask2[i2] = mask2[i2 - 1]
            i2 += 1

        cmp = _compare(arr1[i1], arr2[i2])
        if cmp == 0:
            mask1[i1] = True
            mask2[i2] = True
            i1 += 1
            i2 += 1
        elif cmp == -1:
            i1 += 1
        else:
            i2 += 1

    while 0 < i1 < len1 and _equal(arr1[i1], arr1[i1 - 1]):
        mask1[i1] = mask1[i1 - 1]
        i1 += 1
    while 0 < i2 < len2 and _equal(arr2[i2], arr2[i2 - 1]):
        mask2[i2] = mask2[i2 - 1]
        i2 += 1
    return mask1, mask2


def fast_sorted_overlap(arr1, arr2, return_arr1_indices=True, return_arr2_indices=True):
    """ [Param] arr1 (2D numpy) where row = (i1, i2, ..., ik), or (1D numpy), sorted by all
        [Param] arr2 (2D numpy) where row = (i1, i2, ..., ik), or (1D numpy), sorted by all
        [Return] indices_1_in_2 (1D numpy) sorted
        [Return] indices_1_notin_2 (1D numpy) sorted
        [Return] indices_2_in_1 (1D numpy) sorted
        [Return] indices_2_notin_1 (1D numpy) sorted
    """
    arr1 = np.expand_dims(arr1, 1) if arr1.ndim == 1 else arr1
    arr2 = np.expand_dims(arr2, 1) if arr2.ndim == 1 else arr2
    mask1, mask2 = _fast_sorted_overlap(arr1, arr2)

    results = []
    if return_arr1_indices:
        indices = np.arange(len(arr1))
        indices_1_in_2 = indices[mask1]
        indices_1_notin_2 = indices[np.invert(mask1)]
        results += [indices_1_in_2, indices_1_notin_2]

    if return_arr2_indices:
        indices = np.arange(len(arr2))
        indices_2_in_1 = indices[mask2]
        indices_2_notin_1 = indices[np.invert(mask2)]
        results += [indices_2_in_1, indices_2_notin_1]
    return results


@jit(nopython=True)
def _sorted_matched_pairs(arr, unique_arr):
    """ [Param] arr1 (2D numpy)
        [Param] unique_arr (2D numpy)
    """
    len1, len2 = len(arr), len(unique_arr)
    indices1 = np.empty(len2, dtype=np.int64)
    indices2 = np.empty(len2, dtype=np.int64)
    i1, i2, p = 0, 0, 0
    while i1 < len1 and i2 < len2:
        cmp = _compare(arr[i1], unique_arr[i2])
        if cmp == 0:
            indices1[p] = i1
            indices2[p] = i2
            i1 += 1
            p += 1
        elif cmp == -1:
            i1 += 1
        else:
            i2 += 1

    indices1 = indices1[:p]
    indices2 = indices2[:p]
    return indices1, indices2


def sorted_matched_pairs(arr, unique_arr):
    """ [Param] arr (2D numpy) or (1D numpy), sorted by all
        [Param] unique_arr (2D numpy) or (1D numpy), sorted and unique by all
        [Return] indices1 (1D numpy) sorted
        [Return] indices2 (1D numpy) sorted
    """
    arr = np.expand_dims(arr, 1) if arr.ndim == 1 else arr
    unique_arr = np.expand_dims(unique_arr, 1) if unique_arr.ndim == 1 else unique_arr
    indices1, indices2 = _sorted_matched_pairs(arr, unique_arr)
    return indices1, indices2


@jit(nopython=True)
def _sorted_unique(arr):
    len_ = len(arr)
    indices = np.empty(len_, dtype=np.int64)
    i, p = 0, 0
    while i < len_:
        if i > 0:
            cmp = _compare(arr[i], arr[i-1])
            if cmp != 0:
                indices[p] = i
                p += 1
        else:
            indices[p] = i
            p += 1
        i += 1
    return indices[:p]


def sorted_unique(arr):
    """ [Param] arr (2D numpy) or (1D numpy), sorted by all
        [Return] indices (1D numpy) sorted
    """
    arr = np.expand_dims(arr, 1) if arr.ndim == 1 else arr
    indices = _sorted_unique(arr)
    return indices


def sort(arr, dim=None):
    return arr[argsort(arr, dim=dim)]


def s_unique(arr):
    return arr[sorted_unique(arr)]
