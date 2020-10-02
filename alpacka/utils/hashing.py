"""Hashing utilities"""

from copy import deepcopy
import numpy as np


def _get_hash_key(size):
    state = np.random.get_state()

    np.random.seed(0)
    hash_key = np.random.normal(size=size)

    np.random.set_state(state)
    return hash_key


def _hash_of_np_array(array, hash_key):
    flat_np = array.flatten()
    return int(np.dot(flat_np, hash_key[:len(flat_np)]) * 10e8)


class HashableNdarray:
    """Hashing wrapper for numpy array."""
    hash_key = _get_hash_key(size=10000)

    def __init__(self, array):
        assert isinstance(array, np.ndarray)
        self.array = array
        self._hash = None

    def __hash__(self):
        if self._hash is None:
            self._hash = _hash_of_np_array(
                self.array, HashableNdarray.hash_key
            )
        return self._hash

    def __eq__(self, other):
        return np.array_equal(self.array, other.array)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __deepcopy__(self, memo=None):
        copy = HashableNdarray(
            deepcopy(self.array, memo if memo is not None else {})
        )
        copy._hash = self._hash
        return copy
