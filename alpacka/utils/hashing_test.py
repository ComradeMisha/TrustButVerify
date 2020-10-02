"""Tests for alpacka.utils.hash."""

# pylint: disable=protected-access

import copy

import numpy as np

from alpacka.utils import hashing


def test_basic_hashing_example():
    list1 = [[2, 3],
             [-4, 1]]
    list2 = [[8, -9],
             [-3, -10]]

    harr1 = hashing.HashableNdarray(np.array(list1))
    harr1_copy = hashing.HashableNdarray(np.array(list1))
    harr2 = hashing.HashableNdarray(np.array(list2))

    assert harr1.__hash__() == harr1_copy.__hash__()
    assert harr1.__hash__() != harr2.__hash__()


def test_similar_arrays_hashing():
    shape = (10, 10, 7)

    arr1 = np.zeros(shape)
    arr1[2][3][4] = 1
    hashable1 = hashing.HashableNdarray(arr1)

    arr2 = np.zeros(shape)
    arr2[9][0][5] = 1
    hashable2 = hashing.HashableNdarray(arr2)

    assert hashable1.__hash__() != hashable2.__hash__()


def test_too_big_array():
    """Expect exception on too big arrays.

    Check that hashing raises exception when operating on bigger array
    than the hashing key - we want to fail loudly then.
    """
    arr = np.zeros(10001)  # longer than the hash key size
    hashable = hashing.HashableNdarray(arr)

    raised = False
    try:
        hashable.__hash__()
    except Exception:  # pylint: disable=broad-except
        raised = True

    assert raised


def test_deepcopy():
    """Make sure that copy is independent from original.

    HashableNdarrays shouldn't be modified after construction, but we want to
    make sure that even accidental modification of copy won't affect the
    original.
    """
    arr = np.zeros((4,))
    hashable_orig = hashing.HashableNdarray(arr)
    hashable_orig.__hash__()  # populate hash cache

    hashable_copy = copy.deepcopy(hashable_orig)
    assert np.array_equal(hashable_orig.array, hashable_copy.array)
    assert hashable_orig._hash == hashable_copy._hash

    hashable_copy._hash += 1
    assert hashable_orig._hash != hashable_copy._hash

    hashable_copy.array[2] = 5
    assert not np.array_equal(hashable_orig.array, hashable_copy.array)
