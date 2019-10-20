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


def argsort(arr, dim=None):
    ''' [Param] arr (2D numpy): (i1, i2, ..., ik), or (1D numpy)
        [Return] indices (1D numpy)
    '''
    selected_arr = arr if dim is None else arr[:, dim]
    combo_arr = combo_view(selected_arr)
    indices = np.argsort(combo_arr)
    return indices


def topk_smallest(arr, k):
    ''' [Param] arr (1D numpy) sorted
        [Return] indices (1D numpy) sorted
    '''
    return np.sort(np.argpartition(arr, min(k - 1, len(arr) - 1))[:min(k, len(arr))])


def group_topk(keys, values, k, largest=True):
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


def overlap(arr1, arr2, return_arr1_indices=True, return_arr2_indices=True):
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
        results += [indices_1_in_2, indices_1_notin_2]

    if return_arr2_indices:
        sort_left_1 = combo_arr1.searchsorted(combo_arr2, side='left')
        sort_right_1 = combo_arr1.searchsorted(combo_arr2, side='right')
        indices_2_in_1 = (sort_right_1 - sort_left_1 > 0).nonzero()[0]
        indices_2_notin_1 = (sort_right_1 - sort_left_1 == 0).nonzero()[0]
        results += [indices_2_in_1, indices_2_notin_1]
    return results


def fast_overlap(arr1, arr2, return_arr1_indices=True, return_arr2_indices=True):
    ''' [Param] arr1 (2D numpy): (i1, i2, ..., ik), or (1D numpy), sorted by all
        [Param] arr2 (2D numpy): (i1, i2, ..., ik), or (1D numpy), sorted by all
        [Return] indices_1_in_2
        [Return] indices_1_notin_2
        [Return] indices_2_in_1
        [Return] indices_2_notin_1
    '''
    arr1 = np.expand_dims(arr1, 1) if arr1.ndim == 1 else arr1
    arr2 = np.expand_dims(arr2, 1) if arr2.ndim == 1 else arr2
    mask1, mask2 = _fast_overlap(arr1, arr2)

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
def compare(arr1, arr2):
    ''' [Param] arr1 (1D numpy)
        [Param] arr2 (1D numpy)
        [Note] arr1.shape == arr2.shape
    '''
    for i in range(len(arr1)):
        sign = np.sign(arr1[i] - arr2[i])
        if sign == 0:
            continue
        else:
            return sign
    return 0


@jit(nopython=True)
def equal(arr1, arr2):
    ''' [Param] arr1 (1D numpy)
        [Param] arr2 (1D numpy)
        [Note] arr1.shape == arr2.shape
    '''
    for i in range(len(arr1)):
        if arr1[i] != arr2[i]:
            return False
    return True


@jit(nopython=True)
def _fast_overlap(arr1, arr2):
    ''' [Param] arr1 (2D numpy) or (1D numpy)
        [Param] arr2 (2D numpy) or (1D numpy)
    '''
    len1, len2 = len(arr1), len(arr2)
    mask1 = np.zeros(len1, dtype=np.bool8)
    mask2 = np.zeros(len2, dtype=np.bool8)
    i1, i2 = 0, 0
    while i1 < len1 and i2 < len2:
        while 0 < i1 < len1 and equal(arr1[i1], arr1[i1 - 1]):
            mask1[i1] = mask1[i1 - 1]
            i1 += 1
        while 0 < i2 < len2 and equal(arr2[i2], arr2[i2 - 1]):
            mask2[i2] = mask2[i2 - 1]
            i2 += 1

        cmp = compare(arr1[i1], arr2[i2])
        if cmp == 0:
            mask1[i1] = True
            mask2[i2] = True
            i1 += 1
            i2 += 1
        elif cmp == -1:
            i1 += 1
        else:
            i2 += 1

    while 0 < i1 < len1 and equal(arr1[i1], arr1[i1 - 1]):
        mask1[i1] = mask1[i1 - 1]
        i1 += 1
    while 0 < i2 < len2 and equal(arr2[i2], arr2[i2 - 1]):
        mask2[i2] = mask2[i2 - 1]
        i2 += 1
    return mask1, mask2


if __name__ == '__main__':
    def test_overlap():
        arr1 = np.random.randint(100, size=(10000, 3))
        arr2 = np.random.randint(100, size=(10000, 3))
        arr1 = arr1[argsort(arr1)]
        arr2 = arr2[argsort(arr2)]
        return overlap(arr1, arr2, return_arr1_indices=True, return_arr2_indices=False)


    def test_fast_overlap():
        arr1 = np.random.randint(100, size=(10000, 3))
        arr2 = np.random.randint(100, size=(10000, 3))
        arr1 = arr1[argsort(arr1)]
        arr2 = arr2[argsort(arr2)]
        return fast_overlap(arr1, arr2, return_arr1_indices=True, return_arr2_indices=False)

    from timeit import Timer
    ti = Timer('test_overlap()', 'from __main__ import test_overlap')
    ti2 = Timer('test_fast_overlap()', 'from __main__ import test_fast_overlap')

    print(ti.timeit(1000))
    print(ti2.timeit(1000))
