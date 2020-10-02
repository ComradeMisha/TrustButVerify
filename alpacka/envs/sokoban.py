"""Sokoban environment."""
from copy import deepcopy

import numpy as np
from gym.spaces import Box
from gym.spaces import Discrete
from gym_sokoban.envs import sokoban_env_fast
from gym_sokoban.envs.sokoban_env_fast import load_surfaces

from alpacka.envs import base
from alpacka.envs.base import visualize_transitions
from alpacka.utils.hashing import HashableNdarray


class Sokoban(sokoban_env_fast.SokobanEnvFast):
    """Sokoban with state clone/restore and returning a "solved" flag.

    Returns observations in one-hot encoding.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Return observations as float32, so we don't have to cast them in the
        # network training pipeline.
        self.observation_space.dtype = np.float32

    def reset(self):
        return super().reset().astype(np.float32)

    def step(self, action):
        observation, reward, done, info = super().step(action)
        return observation.astype(np.float32), reward, done, info

    def clone_state(self):
        return self.clone_full_state()

    def restore_state(self, state):
        self.restore_full_state(state)
        return self.render(mode=self.mode)

    def restore_state_from_observation(self, observation):
        self.restore_full_state_from_np_array_version(observation)

    @staticmethod
    def visualize_transitions(transition_batch):
        """Generates visualizations of agent transitions.

        Single visualization is made of true observation at time step t and
        predicted next observations by the TrainableModel.

        Args:
            transition_batch (data.Transition): Transition object containing
                sequence of transitions to be visualized.
        Returns:
            prediction_grids (np.ndarray): Array with RGB images, visualizations
            of TrainableModel predictions at consecutive time steps. It has
            shape of (episode_length, image_H, image_W).
        """
        # Note: we are using TrainableSokoban here, because it can render
        # observations in a stateless way. Unfortunately, such method can not be
        # extracted from Sokoban.
        return visualize_transitions(
            transition_batch, TrainableSokoban.render_observation,
            actions_labels=('up', 'down', 'left', 'right')
        )


class TrainableSokoban(base.TrainableFrameToFrameEnv):
    """Sokoban model based on neural network.

    Returns observations in one-hot encoding.

    While cloning state and restoring state from external objects, we perform
    deep copies in order to avoid accidental modifications of internal state
    by modyfing numpy array. This may cause overhead - get rid of copying
    if it becomes a problem.
    """
    surfaces = load_surfaces()

    def __init__(self, modeled_env=None, dim_room=None, done_threshold=0.5,
                 reward_threshold=0.5):
        super().__init__()
        if modeled_env is not None:
            if dim_room is not None:
                print(
                    'WARN: Both modeled_env and dim_room arguments passed '
                    'to TrainableSokoban - dim_room will be ignored.'
                )
            dim_room = modeled_env.dim_room
        else:
            if dim_room is None:
                raise ValueError(
                    'Either env or dim_room must be provided to '
                    'TrainableSokoban'
                )

        self.action_space = Discrete(4)
        self.observation_space = Box(
            low=np.float32(0), high=np.float32(1),
            shape=(*dim_room, 7), dtype=np.float32
        )
        self.reward_range = (-np.inf, np.inf)
        self._dim_room = dim_room
        self.done_threshold = done_threshold
        self.reward_threshold = reward_threshold

    @staticmethod
    def render_state(state, mode='one_hot'):
        observation = TrainableSokoban.state2obs(state)
        return TrainableSokoban.render_observation(observation, mode)

    @staticmethod
    def render_observation(observation, mode='one_hot'):
        """Renders environment observation."""
        observation = observation.astype(np.uint8)
        if mode == 'one_hot':
            return observation

        if mode == 'rgb_array':
            render_surfaces = TrainableSokoban.surfaces['16x16pixels']
        elif mode == 'tiny_rgb_array':
            render_surfaces = TrainableSokoban.surfaces['8x8pixels']
        else:
            raise ValueError(f'Render mode is not supported: {mode}')

        size_x = observation.shape[0] * render_surfaces.shape[1]
        size_y = observation.shape[1] * render_surfaces.shape[2]

        res = np.tensordot(observation, render_surfaces, (-1, 0))
        res = np.transpose(res, (0, 2, 1, 3, 4))
        res = np.reshape(res, (size_x, size_y, 3))
        return res

    @staticmethod
    def obs2state(observation, copy=True):
        if copy:
            observation = deepcopy(observation)
        return HashableNdarray(observation)

    @staticmethod
    def state2obs(state):
        return state.array
