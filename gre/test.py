from timeit import Timer
import numpy as np
import torch

from gre.fast_numpy import argsort, overlap, fast_overlap
from gre.sparse_ops import none_to_sparse, sparse_index


def test1():
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

    ti = Timer('test_overlap()', 'from __main__ import test_overlap')
    ti2 = Timer('test_fast_overlap()', 'from __main__ import test_fast_overlap')

    print(ti.timeit(1000))
    print(ti2.timeit(1000))


def test2():
    batch_size = 10
    n_nodes = 20
    n_dims = 10
    updated_nodes = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                  [0, 8, 3, 8, 6, 4, 7, 1, 5, 3]])
    states = none_to_sparse((batch_size, n_nodes), (n_dims,))
    print(states)
    states = sparse_index(states, updated_nodes)
    print(states)
    print(states.size())

def test3():
    arr1 = np.random.randint(5, size=(20, 4))
    arr2 = np.random.randint(5, size=(20, 2))
    arr1 = arr1[argsort(arr1)]
    arr2 = arr2[argsort(arr2)]
    print(arr1)
    print(arr2)
    res = fast_overlap(arr1[:,:2], arr2, return_arr1_indices=True, return_arr2_indices=True)
    print(res)

if __name__ == '__main__':
    test3()
