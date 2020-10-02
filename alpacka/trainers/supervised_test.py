"""Tests for alpacka.trainers.supervised."""

import collections

import numpy as np

from alpacka import data
from alpacka import testing
from alpacka.networks import core
from alpacka.networks import keras
from alpacka.trainers import supervised


def test_integration_with_keras():
    TestTransition = collections.namedtuple('TestTransition', ['observation'])

    # Just a smoke test, that nothing errors out.
    n_transitions = 10
    obs_shape = (4,)
    network_sig = data.NetworkSignature(
        input=data.TensorSignature(shape=obs_shape),
        output=data.TensorSignature(shape=(1,)),
    )
    trainer = supervised.SupervisedTrainer(
        network_signature=network_sig,
        target=supervised.target_solved,
        batch_size=2,
        n_steps_per_epoch=3,
        replay_buffer_capacity=n_transitions,
    )
    trainer.add_episode(data.Episode(
        transition_batch=TestTransition(
            observation=np.zeros((n_transitions,) + obs_shape),
        ),
        return_=123,
        solved=False,
    ))
    network = keras.KerasNetwork(network_signature=network_sig)
    trainer.train_epoch(network)


def test_multiple_targets():
    TestTransition = collections.namedtuple(
        'TestTransition', ['observation', 'agent_info']
    )

    network_sig = data.NetworkSignature(
        input=data.TensorSignature(shape=(1,)),
        # Two outputs.
        output=(
            data.TensorSignature(shape=(1,)),
            data.TensorSignature(shape=(2,)),
        ),
    )
    trainer = supervised.SupervisedTrainer(
        network_signature=network_sig,
        # Two targets.
        target=(supervised.target_solved, supervised.target_qualities),
        batch_size=1,
        n_steps_per_epoch=1,
        replay_buffer_capacity=1,
    )
    trainer.add_episode(data.Episode(
        transition_batch=TestTransition(
            observation=np.zeros((1, 1)),
            agent_info={'qualities': np.zeros((1, 2))},
        ),
        return_=123,
        solved=False,
    ))

    class TestNetwork(core.DummyNetwork):
        """Mock class."""

        def train(
            self, data_stream, n_steps, epoch, validation_data_stream=None
        ):
            np.testing.assert_equal(
                list(data_stream()),
                [testing.zero_pytree(
                    (network_sig.input, network_sig.output), shape_prefix=(1,)
                )],
            )

            return {}

    trainer.train_epoch(TestNetwork(network_sig))
