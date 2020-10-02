"""Data transformation utils."""

import numpy as np
import scipy.signal


def discount_cumsum(x, discount):
    """Magic from rllab for computing discounted cumulative sums of vectors.

    Args:
        x (np.array): sequence of floats (eg. rewards from a single episode in
            RL settings)
        discount (float): discount factor (in RL known as gamma)

    Returns:
        Array of cumulative discounted sums. For example:

        If vector x has a form
            [x0,
             x1,
             x2]

        Then the output would be:
            [x0 + discount * x1 + discount^2 * x2,
             x1 + discount * x2,
             x2]
    """
    return scipy.signal.lfilter(
        [1], [1, float(-discount)], x[::-1], axis=0
    )[::-1]


def insert_action_channels(observations, actions, n_actions):
    """Extends observation channels by one-hot encoded action layers."""
    n_observation_channels = observations.shape[-1]
    padding = [(0, 0) for _ in observations.shape]
    padding[-1] = (0, n_actions)
    padded_observations = np.pad(observations, padding)
    for i, action in enumerate(actions):
        padded_observations[i, ..., n_observation_channels + action] = 1

    return padded_observations
