"""Tests for alpacka.envs.sokoban."""

import math

import numpy as np
import tensorflow as tf

from alpacka.data import RequestType
from alpacka.envs.sokoban import TrainableSokoban


def test_trainable_sokoban_predict_steps():
    # Set up Sokoban model
    model = TrainableSokoban(
        dim_room=(2, 2),
        done_threshold=0.95,
        reward_threshold=0.5,
    )

    board_shape = model.observation_space.shape
    board_channels = board_shape[-1]
    n_actions = model.action_space.n
    assert n_actions == 4

    board = tf.one_hot(
        [[1, 2], [3, 4]], depth=board_channels
    ).numpy().astype(np.float32)

    init_state = model.obs2state(board)
    assert np.array_equal(init_state.array, board)

    # Run function and wait for the Request
    actions = list(range(n_actions))
    cor = model.predict_steps(init_state, actions)
    request = cor.send(None)

    # Check if the Request is correct
    base_query = np.pad(board, [(0, 0), (0, 0), (0, n_actions)])
    expected_content = []
    for action in actions:
        query = base_query.copy()
        query[:, :, board_channels + action] = 1
        expected_content.append(query)
    assert request.type == RequestType.MODEL_PREDICTION
    assert np.array_equal(request.content, np.stack(expected_content, axis=0))

    # Prepare expected final results
    one_indices = [(0, 0, 1), (0, 1, 5), (1, 0, 6), (1, 1, 0)]
    expected_observation = np.zeros(shape=board_shape, dtype=np.float32)
    _assign_value_to_indices(expected_observation, one_indices, 1)

    # Prepare response to the Request with mocked network predictions
    highest_value = 0.9
    observation_pred = np.random.uniform(low=0.0, high=0.5, size=board_shape)
    _assign_value_to_indices(observation_pred, one_indices, highest_value)

    observation_preds = np.stack(arrays=[observation_pred] * n_actions)

    reward_preds = np.array([[0.1], [1.0], [0.6], [0.4]])
    expected_rewards = [0, 1, 1, 0]

    done_proba_preds = np.array([[0.1], [1.0], [0.99], [0.9]])
    expected_dones = [False, True, True, False]
    try:
        # Send response
        cor.send({
            'next_observation': observation_preds,
            'reward': reward_preds,
            'done': done_proba_preds
        })
        assert False
    except StopIteration as e:
        observations, rewards, dones, infos, states = e.value

    # Check results
    assert all(
        len(res_list) == n_actions
        for res_list in (observations, rewards, dones, infos, states)
    )
    assert all(
        np.array_equal(observation, expected_observation)
        for observation in observations
    )
    assert all(
        math.isclose(reward, expected_reward)
        for reward, expected_reward in zip(rewards, expected_rewards)
    )
    assert dones == expected_dones
    assert all(info['solved'] == done for info, done in zip(infos, dones))
    assert all(
        np.array_equal(state.array, expected_observation)
        for state in states
    )


def _assign_value_to_indices(array, indices, value):
    for index in indices:
        array[index] = value
