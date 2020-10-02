"""Uniform replay buffer."""

import functools
import itertools
import random

import gin
import numpy as np
import randomdict
from scipy.stats import multinomial

from alpacka import data


class UniformReplayBuffer:
    """Replay buffer with uniform sampling.

    Stores datapoints in a queue of fixed size. Adding to a full buffer
    overwrites the oldest ones.
    """

    def __init__(self, datapoint_signature, capacity):
        """Initializes the replay buffer.

        Args:
            datapoint_signature (pytree): Pytree of TensorSignatures, defining
                the structure of data to be stored.
            capacity (int): Maximum size of the buffer.
        """
        self._capacity = int(capacity)
        self._size = 0
        self._insert_index = 0

        def init_array(signature):
            shape = (self._capacity,) + signature.shape
            return np.zeros(shape, dtype=signature.dtype)
        self._data_buffer = data.nested_map(init_array, datapoint_signature)

    def __iter__(self):
        return (
            data.nested_map(lambda x, idx=index: x[idx], self._data_buffer)
            for index in range(self._size)
        )

    def __len__(self):
        return self._size

    @property
    def use_importance_sampling(self):
        return None

    def add(self, stacked_datapoints):
        """Adds datapoints to the buffer.

        Args:
            stacked_datapoints (pytree): Transition object containing the
                datapoints, stacked along axis 0.
        """
        n_elems = data.choose_leaf(data.nested_map(
            lambda x: x.shape[0], stacked_datapoints
        ))

        def insert_to_array(buf, elems):
            buf_size = buf.shape[0]
            assert elems.shape[0] == n_elems
            index = self._insert_index
            # Insert up to buf_size at the current index.
            buf[index:min(index + n_elems, buf_size)] = elems[:buf_size - index]
            # Insert whatever's left at the beginning of the buffer.
            buf[:max(index + n_elems - buf_size, 0)] = elems[buf_size - index:]

        # Insert to all arrays in the pytree.
        data.nested_zip_with(
            insert_to_array, (self._data_buffer, stacked_datapoints)
        )
        if self._size < self._capacity:
            self._size = min(self._insert_index + n_elems, self._capacity)
        self._insert_index = (self._insert_index + n_elems) % self._capacity

    def sample(self, batch_size):
        """Samples a batch of datapoints.

        Args:
            batch_size (int): Number of datapoints to sample.

        Returns:
            Datapoint object with sampled datapoints stacked along the 0 axis.

        Raises:
            ValueError: If the buffer is empty.
        """
        if self._data_buffer is None:
            raise ValueError('Cannot sample from an empty buffer.')
        indices = np.random.randint(self._size, size=batch_size)
        return data.nested_map(lambda x: x[indices], self._data_buffer)

    def update(self, idx, error):
        pass

    def top_priority_samples(self, n, top=0.95):
        # Hack: return empty list of samples and priorities (and not n ones)
        del n, top
        return [], []


@gin.configurable
class PrioritizedReplayBuffer:
    """Replay buffer with priority sampling."""
    def __init__(self, datapoint_signature, capacity, **kwargs):
        """Initializes the replay buffer.

        Args:
            datapoint_signature (pytree): Ignored, left for compatibility.
            capacity (int): Maximum size of the buffer.
            **kwargs: Arguments passed to Memory.
        """
        del datapoint_signature
        self.memory = Memory(int(capacity), **kwargs)

    def __len__(self):
        return len(self.memory)

    @property
    def use_importance_sampling(self):
        return self.memory.use_importance_sampling

    def add(self, stacked_datapoints):
        """Adds datapoints to the buffer.

        Args:
            stacked_datapoints (pytree): Transition object containing the
                datapoints, stacked along axis 0.
        """
        for datapoint in data.nested_unstack(stacked_datapoints):
            self.memory.add(datapoint)

    def sample(self, batch_size):
        batch, idxs, is_weights = self.memory.sample(batch_size)
        x, y = data.nested_stack(batch)
        is_weights_per_output = {
            output_name: np.expand_dims(is_weights, axis=-1)
            for output_name in y.keys()
        }
        idxs = np.expand_dims(idxs, axis=-1)
        return x, y, idxs, is_weights_per_output

    def update(self, idx, error):
        self.memory.update(idx, error)

    def mean_priority(self):
        return self.memory.mean_priority()

    def top_priority_samples(self, n, top=0.95):
        return self.memory.top_priority_samples(n, top)


class HierarchicalReplayBuffer:
    """Replay buffer with hierarchical sampling.

    Datapoints are indexed by a list of "buckets". Buckets are sampled
    uniformly, in a fixed order. Each sequence of buckets has its own capacity.
    """

    def __init__(
            self,
            datapoint_sig,
            capacity,
            hierarchy_depth,
            raw_buffer=UniformReplayBuffer,
    ):
        """Initializes HierarchicalReplayBuffer.

        Args:
            datapoint_sig (pytree): Pytree of TensorSignatures, defining
                the structure of data to be stored.
            capacity (int): Maximum size of the buffer.
            hierarchy_depth (int): Number of buckets in the hierarchy.
            raw_buffer (Optional[Callable[datapoint_sig, int]]): Raw buffer used
                for each bucket.
        """
        self._raw_buffer_fn = functools.partial(
            raw_buffer, datapoint_sig, capacity
        )
        # Data is stored in a tree, where inner nodes are dicts with fast
        # random sampling (RandomDicts) and leaves are UniformReplayBuffers.
        # This won't scale to buckets with a large number of possible values.
        self._hierarchy_depth = hierarchy_depth
        if self._hierarchy_depth:
            self._buffer_hierarchy = randomdict.RandomDict()
        else:
            self._buffer_hierarchy = self._raw_buffer_fn()

    def __iter__(self):
        if self._hierarchy_depth > 1:
            raise NotImplementedError('Hierarchy depth > 1 not supported.')
        if self._hierarchy_depth:
            iterable_buffers = [
                iter(bucket_buffer) for bucket, bucket_buffer
                in self._buffer_hierarchy.values
            ]
        else:
            iterable_buffers = [iter(self._buffer_hierarchy)]

        # We iterate one buffer at the time. It may cause problems when used for
        # training purposes, as batches will be biased.
        # Currently we are using this only for validation replay buffers, for
        # which it does not matter, as we gather all statistics and the end
        # anyway.
        return itertools.chain(*iterable_buffers)

    @property
    def use_importance_sampling(self):
        buffer_hierarchy = self._buffer_hierarchy
        for _ in range(self._hierarchy_depth):
            buffer_hierarchy = buffer_hierarchy.random_value()
        return buffer_hierarchy.use_importance_sampling

    def add(self, stacked_datapoints, buckets):
        """Adds datapoints to the buffer.

        Args:
            stacked_datapoints (pytree): Transition object containing the
                datapoints, stacked along axis 0.
            buckets (list): List of length hierarchy_depth with values of
                buckets.
        """
        assert len(buckets) == self._hierarchy_depth

        # Because Python doesn't support value assignment, we need to if out
        # the case when the hierarchy is flat.
        if self._hierarchy_depth:
            buffer_hierarchy = self._buffer_hierarchy
            for bucket in buckets[:-1]:
                if bucket not in buffer_hierarchy:
                    buffer_hierarchy[bucket] = randomdict.RandomDict()
                buffer_hierarchy = buffer_hierarchy[bucket]

            bucket = buckets[-1]
            if bucket not in buffer_hierarchy:
                buffer_hierarchy[bucket] = self._raw_buffer_fn()
            buf = buffer_hierarchy[bucket]
        else:
            buf = self._buffer_hierarchy

        buf.add(stacked_datapoints)

    def _sample_one(self):
        buffer_hierarchy = self._buffer_hierarchy
        for _ in range(self._hierarchy_depth):
            buffer_hierarchy = buffer_hierarchy.random_value()
        return buffer_hierarchy.sample(batch_size=1)

    def sample(self, batch_size, buckets=False):
        """Samples a batch of datapoints.

        Args:
            batch_size (int): Number of datapoints to sample.
            buckets (bool): Indicates if buckets indices should be returned.

        Returns:
            Datapoint object with sampled datapoints stacked along the 0 axis.

        Raises:
            ValueError: If the buffer is empty.
        """
        if self._hierarchy_depth > 1:
            samples = [self._sample_one() for _ in range(batch_size)]
        else:
            p = np.ones((len(self._buffer_hierarchy),), dtype=np.float32)
            p = np.atleast_1d(p) / p.sum()
            samples = []
            distribution = multinomial(batch_size, p=p)
            rvs = distribution.rvs(1).squeeze(axis=0)
            for bucket, n_samples in zip(self._buffer_hierarchy, rvs):
                if self._hierarchy_depth > 0:
                    buffer = self._buffer_hierarchy[bucket]
                else:
                    buffer = self._buffer_hierarchy
                samples_b = buffer.sample(n_samples)
                if buckets:
                    samples_b = samples_b + (np.full(n_samples, bucket),)
                samples.append(samples_b)

        return data.nested_concatenate(samples)

    def buffers_sizes(self):
        if self._hierarchy_depth > 0:
            return {
                bucket: len(buffer)
                for bucket, buffer in self._buffer_hierarchy.items()
            }
        else:
            return {
                'default': len(self._buffer_hierarchy)
            }

    def mean_priorities(self):
        return {
            bucket: buffer.mean_priority()
            for bucket, buffer in self._buffer_hierarchy.items()
        }

    def top_priority_samples(self, n, top=0.95):
        return {
            bucket: buffer.top_priority_samples(n, top)
            for bucket, buffer in self._buffer_hierarchy.items()
        }

    def update(self, idx, bucket_idx, error):
        self._buffer_hierarchy[bucket_idx].update(idx, error)

    def clear(self):
        if self._hierarchy_depth:
            self._buffer_hierarchy = randomdict.RandomDict()
        else:
            self._buffer_hierarchy = self._raw_buffer_fn()
