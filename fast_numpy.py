from numba import jit
import numpy as np


def combo_view(arr):
    ''' [Param] arr (2D numpy) or (1D numpy)
        [Return] combo_arr (1D numpy)
    '''
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


def anti_combo_view(combo_arr, orig_arr):
    arr = combo_arr.view(orig_arr.dtype)
    if orig_arr.ndim == 1:
        return arr
    elif orig_arr.ndim == 2:
        return arr.reshape(-1, orig_arr.shape[1])
    else:
        raise ValueError('`orig_arr` must be a 1D or 2D numpy array')


def sort(arr, dim=None, return_values=False):
    ''' [Param] arr (2D numpy): (i1, i2, ..., ik), or (1D numpy)
        [Return] indices (1D numpy)
        [Return] sorted_arr (2D numpy): (i1, i2, ..., ik) sorted by all or (i[dim[0]), i[dim[1]], ..., i[dim[l]])
    '''
    selected_arr = arr if dim is None else arr[:, dim]
    combo_arr = combo_view(selected_arr)
    indices = np.argsort(combo_arr)
    if return_values:
        sorted_arr = arr[indices]
        return indices, sorted_arr
    else:
        return indices


def topk_smallest(arr, k):
    ''' [Param] arr (1D numpy) sorted
        [Return] indices (1D numpy) sorted
    '''
    return np.sort(np.argpartition(arr, min(k - 1, len(arr) - 1))[:min(k, len(arr))])


def group_topk(keys, values, k, largest=True, return_values=False):
    ''' [Param] keys (2D numpy): (i1, i2, ..., ik) sorted by all, or (1D numpy) sorted
        [Param] values (2D numpy): (v1, v2, ..., vl), or (1D numpy)
        [Return] indices (1D numpy) sorted
    '''
    combo_keys = combo_view(keys)
    combo_values = combo_view(-values if largest else values)
    group_mask = (combo_keys[1:] != combo_keys[:-1])
    n_rows = len(combo_keys)
    group_partition = np.concatenate([np.array([0]), np.arange(1, n_rows)[group_mask], np.array([n_rows])])
    indices = np.concatenate([beg + topk_smallest(combo_values[beg:end], k)
                              for beg, end in zip(group_partition[:-1], group_partition[1:])])
    if return_values:
        return indices, keys[indices], values[indices]
    else:
        return indices


def intersect(arr1, arr2, assume_unique=False, return_values=False):
    ''' [Param] arr1 (2D numpy): (i1, i2, ..., ik), or (1D numpy)
        [Param] arr2 (2D numpy): (i1, i2, ..., ik), or (1D numpy)
        [Return] indices1: only return the indices of the first occurrences of the common values in arr1
        [Return] indices2: only return the indices of the first occurrences of the common values in arr2
        [Return] intersected (2D numpy): (i1, i2, ..., ik) sorted and unique
    '''
    combo_arr1 = combo_view(arr1)
    combo_arr2 = combo_view(arr2)
    intersected, indices1, indices2 = np.intersect1d(combo_arr1, combo_arr2,
                                                     assume_unique=assume_unique, return_indices=True)
    if return_values:
        intersected = anti_combo_view(intersected, arr1)
        return indices1, indices2, intersected
    else:
        return indices1, indices2


def overlap(arr1, arr2, return_arr1_indices=True, return_arr2_indices=True, return_values=False):
    ''' [Param] arr1 (2D numpy): (i1, i2, ..., ik), or (1D numpy), sorted by all
        [Param] arr2 (2D numpy): (i1, i2, ..., ik), or (1D numpy), sorted by all
        [Return] indices_1_in_2
        [Return] indices_1_notin_2
        [Return] indices_2_in_1
        [Return] indices_2_notin_1
    '''
    combo_arr1 = combo_view(arr1)
    combo_arr2 = combo_view(arr2)

    results = []
    if return_arr1_indices:
        sort_left_2 = combo_arr2.searchsorted(combo_arr1, side='left')
        sort_right_2 = combo_arr2.searchsorted(combo_arr1, side='right')
        indices_1_in_2 = (sort_right_2 - sort_left_2 > 0).nonzero()[0]
        indices_1_notin_2 = (sort_right_2 - sort_left_2 == 0).nonzero()[0]
        if return_values:
            results += [indices_1_in_2, arr1[indices_1_in_2], indices_1_notin_2, arr1[indices_1_notin_2]]
        else:
            results += [indices_1_in_2, indices_1_notin_2]

    if return_arr2_indices:
        sort_left_1 = combo_arr1.searchsorted(combo_arr2, side='left')
        sort_right_1 = combo_arr1.searchsorted(combo_arr2, side='right')
        indices_2_in_1 = (sort_right_1 - sort_left_1 > 0).nonzero()[0]
        indices_2_notin_1 = (sort_right_1 - sort_left_1 == 0).nonzero()[0]
        if return_values:
            results += [indices_2_in_1, arr2[indices_2_in_1], indices_2_notin_1, arr2[indices_2_notin_1]]
        else:
            results += [indices_2_in_1, indices_2_notin_1]
    return results


def overlap_v2(arr1, arr2, assume_unique=False, return_arr1_indices=True, return_arr2_indices=True, return_values=False):
    ''' [Param] arr1 (2D numpy): (i1, i2, ..., ik), or (1D numpy)
        [Param] arr2 (2D numpy): (i1, i2, ..., ik), or (1D numpy)
        [Return] indices_1_in_2
        [Return] indices_1_notin_2
        [Return] indices_2_in_1
        [Return] indices_2_notin_1
        [Note] slower than overlap()
    '''
    combo_arr1 = combo_view(arr1)
    combo_arr2 = combo_view(arr2)

    results = []
    if return_arr1_indices:
        mask = np.isin(combo_arr1, combo_arr2, assume_unique=assume_unique)
        indices = np.arange(len(mask))
        indices_1_in_2 = indices[mask]
        indices_1_notin_2 = indices[np.invert(mask)]
        if return_values:
            results += [indices_1_in_2, arr1[indices_1_in_2], indices_1_notin_2, arr1[indices_1_notin_2]]
        else:
            results += [indices_1_in_2, indices_1_notin_2]

    if return_arr2_indices:
        mask = np.isin(combo_arr2, combo_arr1, assume_unique=assume_unique)
        indices = np.arange(len(mask))
        indices_2_in_1 = indices[mask]
        indices_2_notin_1 = indices[np.invert(mask)]
        if return_values:
            results += [indices_2_in_1, arr2[indices_2_in_1], indices_2_notin_1, arr2[indices_2_notin_1]]
        else:
            results += [indices_2_in_1, indices_2_notin_1]
    return results


def fast_overlap(combo_arr1, combo_arr2, return_arr1_indices=True, return_arr2_indices=True, return_values=False):
    ''' [Param] combo_arr1 (2D numpy): (i1, i2, ..., ik), or (1D numpy), sorted by all
        [Param] combo_arr2 (2D numpy): (i1, i2, ..., ik), or (1D numpy), sorted by all
    '''
    len1 = len(combo_arr1)
    len2 = len(combo_arr2)
    mask1 = np.zeros(len1, dtype=np.bool8)
    mask2 = np.zeros(len2, dtype=np.bool8)
    i1, i2 = 0, 0
    while i1 < len1 and i2 < len2:
        if combo_arr1[i1] == combo_arr2[i2]:
            mask1[i1] = True
            mask2[i2] = True
        elif combo_arr1[i1] < combo_arr2[i2]:
            i1 += 1
        else:
            i2 += 1

    results = []
    if return_arr1_indices:
        indices = np.arange(len1)
        indices_1_in_2 = indices[mask1]
        indices_1_notin_2 = indices[np.invert(mask1)]
        if return_values:
            results += [indices_1_in_2, combo_arr1[indices_1_in_2], indices_1_notin_2, combo_arr1[indices_1_notin_2]]
        else:
            results += [indices_1_in_2, indices_1_notin_2]

    if return_arr2_indices:
        indices = np.arange(len2)
        indices_2_in_1 = indices[mask2]
        indices_2_notin_1 = indices[np.invert(mask2)]
        if return_values:
            results += [indices_2_in_1, combo_arr2[indices_2_in_1], indices_2_notin_1, combo_arr2[indices_2_notin_1]]
        else:
            results += [indices_2_in_1, indices_2_notin_1]
    return results


@jit(nopython=True)
def _fast_overlap(arr1, arr2):
    len1, len2 = len(arr1), len(arr2)
    mask1 = np.zeros(len1, dtype=np.bool8)
    mask2 = np.zeros(len2, dtype=np.bool8)
    i1, i2 = 0, 0
    while i1 < len1 and i2 < len2:
        t1 = (int(v) for v in arr1[i1])
        t2 = (int(v) for v in arr2[i2])
        if t1 == t2:
            mask1[i1] = True
            mask2[i2] = True
            i1 += 1
            i2 += 1
        elif t1 < t2:
            i1 += 1
        else:
            i2 += 1
    return mask1, mask2

def test_v1():
    arr1 = np.random.randint(100, size=(10000, 3))
    arr2 = np.random.randint(100, size=(10000, 3))
    ind_1_in_2, ind_1_notin_2 = overlap(arr1, arr2, return_arr1_indices=True, return_arr2_indices=False,
                                        return_values=False)


def test_v2():
    arr1 = np.random.randint(100, size=(10000, 3))
    arr2 = np.random.randint(100, size=(10000, 3))
    ind_1_in_2, ind_1_notin_2 = overlap_v2(arr1, arr2, return_arr1_indices=True, return_arr2_indices=False,
                                           return_values=False)

if __name__ == '__main__':
    arr1 = np.random.randint(10, size=(100, 3))
    _, arr1 = sort(arr1, return_values=True)
    combo_arr1 = combo_view(arr1)

    arr2 = np.random.randint(10, size=(100, 3))
    _, arr2 = sort(arr2, return_values=True)
    combo_arr2 = combo_view(arr2)

    print(arr1)
    print(arr2)
    print(_fast_overlap(arr1, arr2))
