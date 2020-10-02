"""Supervised trainer."""
from functools import partial

import gin
import numpy as np

from alpacka import data
from alpacka.trainers import base
from alpacka.trainers import replay_buffers
from alpacka.utils.transformations import insert_action_channels


@gin.configurable
def target_solved(episode):
    return np.full(
        shape=(episode.transition_batch.observation.shape[:1] + (1,)),
        fill_value=int(episode.solved),
    )


@gin.configurable
def target_return(episode):
    return np.cumsum(episode.transition_batch.reward[::-1],
                     dtype=np.float)[::-1, np.newaxis]


@gin.configurable
def target_discounted_return(episode):
    """Uses discounted_return calculated by agent."""
    return np.expand_dims(
        episode.transition_batch.agent_info['discounted_return'], axis=1
    )


@gin.configurable
def target_value(episode):
    return np.expand_dims(
        episode.transition_batch.agent_info['value'], axis=1
    )


@gin.configurable
def target_qualities(episode):
    return episode.transition_batch.agent_info['qualities']


@gin.configurable
def target_action_histogram(episode):
    return episode.transition_batch.agent_info['action_histogram']


@gin.configurable
def target_action_histogram_smooth(episode):
    return episode.transition_batch.agent_info['action_histogram_smooth']


@gin.configurable
def target_model(episode):
    dones = np.logical_and(episode.transition_batch.done, not episode.truncated)
    return {
        'next_observation': episode.transition_batch.next_observation,
        'reward': np.expand_dims(episode.transition_batch.reward, axis=1),
        'done': np.expand_dims(dones, axis=1)
    }


@gin.configurable
def target_model_delta(episode):
    next_obs_delta = (
        episode.transition_batch.next_observation -
        episode.transition_batch.observation
    )
    dones = np.logical_and(episode.transition_batch.done, not episode.truncated)
    return {
        'next_observation': next_obs_delta,
        'reward': np.expand_dims(episode.transition_batch.reward, axis=1),
        'done': np.expand_dims(dones, axis=1)
    }


@gin.configurable
def input_observation(episode):
    return episode.transition_batch.observation


@gin.configurable
def input_observation_and_action(episode, env_class):
    actions_space_size = env_class().action_space.n
    return insert_action_channels(
        episode.transition_batch.observation,
        episode.transition_batch.action,
        actions_space_size
    )


class SupervisedTrainer(base.Trainer):
    """Supervised trainer.

    Trains the network based on (x, y) pairs generated out of transitions
    sampled from a replay buffer.
    """

    def __init__(
        self,
        network_signature,
        inputs=input_observation,
        target=target_solved,
        batch_size=64,
        n_steps_per_epoch=1000,
        replay_buffer_capacity=1000000,
        replay_buffer_sampling_hierarchy=(),
        validation_split=None,
        validate_every_n_epochs=None,
        validation_replay_buffer_capacity=None,
    ):
        """Initializes SupervisedTrainer.

        Args:
            network_signature (pytree): Input signature for the network.
            inputs (callable): Function Episode -> Datapoint_signature.input.
                Preprocesses episodes to the network inputs for further
                training.
            target (pytree): Pytree of functions episode -> target for
                determining the targets for network training. The structure of
                the tree should reflect the structure of a target.
            batch_size (int): Batch size.
            n_steps_per_epoch (int): Number of optimizer steps to do per
                epoch.
            replay_buffer_capacity (int): Maximum size of the replay buffer.
            replay_buffer_sampling_hierarchy (tuple): Sequence of Episode
                attribute names, defining the sampling hierarchy.
            validation_split (Optional[float]): Fraction of episodes which
                should be placed in the validation replay buffer.
            validate_every_n_epochs (Optional[int]): Validation frequency in
                epochs.
            validation_replay_buffer_capacity (Optional[int]): Maximum size
                of the validation replay buffer. Defaults to
                `replay_buffer_capacity` if not provided.

        Raises:
            ValueError: When validation parameters are provided only partially.
        """
        super().__init__(network_signature)
        self._input_fn = inputs
        self._target_fn = lambda episode: data.nested_map(
            lambda f: f(episode), target
        )
        self._batch_size = batch_size
        self._n_steps_per_epoch = n_steps_per_epoch

        # (input, target)
        datapoint_sig = (network_signature.input, network_signature.output)
        self._replay_buffer = replay_buffers.HierarchicalReplayBuffer(
            datapoint_sig,
            capacity=replay_buffer_capacity,
            hierarchy_depth=len(replay_buffer_sampling_hierarchy),
        )
        self._sampling_hierarchy = replay_buffer_sampling_hierarchy

        self._validation_split = validation_split
        self._validate_every_n_epochs = validate_every_n_epochs
        self._validation_replay_buffer = None
        if self._validation_split is not None:
            if self._validate_every_n_epochs is None:
                raise ValueError(
                    f'Argument validate_every_n_epochs should be specified '
                    f'when validation_split is provided: {validation_split}')
            if self._validate_every_n_epochs <= 0:
                raise ValueError(
                    f'Argument validate_every_n_epochs should be positive '
                    f'integer, got {validate_every_n_epochs}')
            self._validation_replay_buffer = \
                replay_buffers.HierarchicalReplayBuffer(
                    datapoint_sig,
                    validation_replay_buffer_capacity or replay_buffer_capacity,
                    len(self._sampling_hierarchy)
                )
        self._epoch = 0

    def add_episode(self, episode, auxiliary_bucket_setting=None):
        buckets = self._get_bucket_hierarchy(episode, auxiliary_bucket_setting)
        target_buffer = self._replay_buffer
        if (self._validation_replay_buffer is not None and
                np.random.rand() < self._validation_split):
            target_buffer = self._validation_replay_buffer

        target_buffer.add(
            (
                self._input_fn(episode),  # input
                self._target_fn(episode),  # target
            ),
            buckets,
        )

    def train_epoch(self, network):
        def training_data_stream():
            for _ in range(self._n_steps_per_epoch):
                yield self._replay_buffer.sample(self._batch_size)

        validation_data_stream = None
        if (self._validation_replay_buffer is not None and
                self._epoch % self._validate_every_n_epochs == 0):
            validation_data_stream = partial(
                self._data_stream_from_replay_buffer,
                self._validation_replay_buffer,
                self._batch_size,
                drop_remainder=False
            )
        metrics = network.train(
            training_data_stream, n_steps=self._n_steps_per_epoch,
            epoch=self._epoch, validation_data_stream=validation_data_stream
        )
        self._epoch += 1
        sizes = self._replay_buffer.buffers_sizes()
        metrics.update({
            f'bucket_{bucket}_size': size for bucket, size in sizes.items()
        })
        return metrics

    def _get_bucket_hierarchy(self, episode, auxiliary_bucket_setting):
        if auxiliary_bucket_setting is None:
            auxiliary_bucket_setting = dict()
        buckets = []
        for bucket_name in self._sampling_hierarchy:
            if hasattr(episode, bucket_name):
                bucket = getattr(episode, bucket_name)
            elif bucket_name in auxiliary_bucket_setting:
                bucket = auxiliary_bucket_setting[bucket_name]
            else:
                raise ValueError(
                        f'Could not determine a bucket with name {bucket_name} '
                        'for the episode.'
                )
            buckets.append(bucket)

        return buckets

    @staticmethod
    def _data_stream_from_replay_buffer(replay_buffer,
                                        batch_size,
                                        drop_remainder=False):
        data_iterator = iter(replay_buffer)
        batch_samples = []
        try:
            while True:
                for _ in range(batch_size):
                    next_el = next(data_iterator)
                    batch_samples.append(next_el)
                yield data.nested_stack(batch_samples)
                batch_samples = []
        except StopIteration:
            if not drop_remainder and len(batch_samples) > 0:
                yield data.nested_stack(batch_samples)

    def clear_experience(self):
        self._replay_buffer.clear()


class SupervisedPriorityTrainer(SupervisedTrainer):
    """Supervised trainer with Priority Experience Replay."""
    def __init__(
            self,
            network_signature,
            replay_buffer_capacity=1000000,
            replay_buffer_sampling_hierarchy=(),
            sample_mode='priority',
            **kwargs
    ):
        super().__init__(
            network_signature,
            replay_buffer_capacity=replay_buffer_capacity,
            replay_buffer_sampling_hierarchy=replay_buffer_sampling_hierarchy,
            **kwargs
        )
        self._sample_mode = sample_mode
        if sample_mode == 'priority':
            raise NotImplementedError("Prioritized replay buffers have been removed.")
        elif sample_mode == 'uniform':
            raw_buffer = replay_buffers.UniformReplayBuffer
        else:
            raise ValueError(
                f'Sample mode {sample_mode} is not supported. Please use '
                f'"priority" or "uniform" sampling.')
        self._replay_buffer = replay_buffers.HierarchicalReplayBuffer(
            network_signature, replay_buffer_capacity,
            hierarchy_depth=len(replay_buffer_sampling_hierarchy),
            raw_buffer=raw_buffer
        )
        if sample_mode == 'priority':
            def mse(y_pred, y_true):
                loss = (y_pred - y_true) ** 2
                return loss.mean(axis=1)
            self._priority_error_fn = mse
        else:
            self._priority_error_fn = None

    @property
    def replay_sample_mode(self):
        return self._sample_mode

    def top_priority_samples(self, n, top=0.95):
        if self._sample_mode == 'priority':
            return self._replay_buffer.top_priority_samples(n, top)
        else:
            raise TypeError(
                f'Trying to sample top priority transitions from replay buffer '
                f'with sample mode {self._sample_mode}.'
            )

    def _train_epoch_with_priority_sampling(self, network):
        metrics = None
        for i in range(self._n_steps_per_epoch):
            samples = self._replay_buffer.sample(self._batch_size, buckets=True)
            x, y, idxs, importance_sampling_weights, bucket_idxs = samples
            if self._replay_buffer.use_importance_sampling:
                importance_sampling_weights = {
                    names: np.squeeze(values, axis=-1)
                    for names, values in importance_sampling_weights.items()
                }
            else:
                importance_sampling_weights = None
            metrics = network.train_one_step(
                x, y, importance_sampling_weights, reset_metrics=(i == 0)
            )
            y_pred = network.predict(x)
            errors = self._priority_error_fn(
                y['next_observation'], y_pred['next_observation']
            )
            for idx, bucket_idx, error in zip(idxs, bucket_idxs, errors):
                self._replay_buffer.update(idx, bucket_idx, error)

        self._epoch += 1
        sizes = self._replay_buffer.buffers_sizes()
        metrics.update({
            f'bucket_{bucket}_size': size for bucket, size in sizes.items()
        })
        mean_priorities = self._replay_buffer.mean_priorities()
        metrics.update({
            f'bucket_{bucket}_mean_priority': mean_priority
            for bucket, mean_priority in mean_priorities.items()
        })
        return metrics

    def train_epoch(self, network):
        if self._sample_mode == 'uniform':
            metrics = super(SupervisedPriorityTrainer, self).train_epoch(
                network)
        elif self._sample_mode == 'priority':
            metrics = self._train_epoch_with_priority_sampling(network)
        else:
            raise ValueError(f'Unknown sampling mode "{self._sample_mode}"')
        return metrics
