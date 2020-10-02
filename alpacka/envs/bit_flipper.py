"""Simple Bit flipper gym.
Based on: https://github.com/RobertTLange/gym-hanoi
"""
from copy import deepcopy

import pygame
from gym import spaces
import numpy as np

from alpacka.envs import base
from alpacka.utils.hashing import HashableNdarray

#These constant are used for episode rendering
BACKGROUND_COLOR = (255, 255, 255)
BIT_SIZE = 14
BIT_SEPARATOR = 2
ZERO_COLOR = (0, 0, 255)
ONE_COLOR = (255, 0, 0)
ACTION_COLOR = (0, 255, 255)
FIRST_FLIP_COLOR = (0, 255, 0)

class BitFlipper(base.ModelEnv):
    """Bit flipper env adhere to Open AI gym template"""
    metadata = {'render.modes': ['human']}

    def __init__(self, n_bits=None, reward_for_solved=1.,
                 reward_for_invalid_action=0):
        self.n_bits = n_bits
        self._reward_for_solved = reward_for_solved
        self._reward_for_invalid_action = reward_for_invalid_action
        self.action_space = spaces.Discrete(self.n_bits)
        self.observation_space = \
            spaces.Box(low=np.array([0] * self.n_bits),
                       high=np.array([1] * self.n_bits),
                       dtype=np.uint8
                       )

        self._current_state = None
        self.goal_state = self.n_bits * (1,)
        self.done = None

    def step(self, action):
        def flip_bit(bit_i):
            self._current_state = tuple(self._current_state[i] if i != bit_i
                                        else (self._current_state[i] + 1) % 2
                                        for i in range(self.n_bits))

        if self.done:
            raise RuntimeError('Episode has finished. '
                               'Call env.reset() to start a new episode.')

        info = {'invalid_action': False}
        if self.move_allowed(action):
            flip_bit(action)
        else:
            info['invalid_action'] = True

        if self._current_state == self.goal_state:
            reward = self._reward_for_solved
            info['solved'] = True
            self.done = True
        elif info['invalid_action']:
            reward = self._reward_for_invalid_action
        else:
            reward = 0

        return self.vectorized_obs(), reward, self.done, info

    def move_allowed(self, action):
        if self._current_state[action] == 1:
            return True
        elif np.all(self._current_state[:action]):
            return True
        else:
            return False

    def clone_state(self):
        return HashableNdarray(np.array(self._current_state, dtype=np.uint8))

    def restore_state(self, state):
        self._current_state = tuple(state.array)
        self.done = self._current_state == self.goal_state

    def obs2tuple(self, obs):
        return tuple(obs)

    @staticmethod
    def state2obs(state):
        return state.array

    @staticmethod
    def obs2state(observation, copy=True):
        if copy:
            observation = deepcopy(observation)
        return HashableNdarray(observation)

    def vectorized_obs(self):
        return np.array(self._current_state)

    def reset(self):
        self._current_state = self.n_bits * (0,)
        self.done = False
        return self.vectorized_obs()

    def render(self, mode='human'):
        return str(self._current_state)

    def visualize_transitions(self, transition_batch):
        """Renders BitFlipper episode as one picture"""
        bit_color_scheme = {0: ZERO_COLOR, 1: ONE_COLOR}
        bit_moved_before = set()
        pygame.init() # pylint: disable=no-member
        map_ = pygame.Surface( # pylint: disable=too-many-function-args
            (BIT_SIZE * len(transition_batch.next_observation),
             2 * (BIT_SIZE + 1) * self.n_bits,)
        )
        map_.fill(BACKGROUND_COLOR)
        for row, obs in enumerate(transition_batch.next_observation):
            for col, bit in enumerate(obs):
                rec = pygame.Rect(
                    (row + 1) * BIT_SIZE + BIT_SEPARATOR,
                    col * BIT_SIZE + BIT_SEPARATOR,
                    BIT_SIZE-BIT_SEPARATOR,
                    BIT_SIZE-BIT_SEPARATOR
                )
                bit_color = bit_color_scheme[bit]
                if bit == 1 and col not in bit_moved_before:
                    bit_moved_before.add(col)
                    bit_color = FIRST_FLIP_COLOR
                pygame.draw.rect(map_, bit_color, rec)

        #draw actions:
        for row, action in enumerate(transition_batch.action):
            rec = pygame.Rect((row + 1) * BIT_SIZE + BIT_SEPARATOR,
                              action * BIT_SIZE + BIT_SEPARATOR,
                              BIT_SIZE - BIT_SEPARATOR,
                              BIT_SIZE - BIT_SEPARATOR)
            pygame.draw.rect(map_, ACTION_COLOR, rec, BIT_SEPARATOR)

        #draw predicted observations:
        if 'children_observations' in transition_batch.agent_info:
            predicted_next_observations_i = np.empty_like(
                transition_batch.next_observation
            )
            predicted_rewards_i = np.empty_like(
                transition_batch.reward
            )
            for t, (action_t, observations_t, rewards_t) in enumerate(zip(
                    transition_batch.action,
                    transition_batch.agent_info[
                        'children_observations'],
                    transition_batch.agent_info['children_rewards']
            )):
                predicted_next_observations_i[t] = observations_t[action_t]
                predicted_rewards_i[t] = rewards_t[action_t]

            for row, obs in enumerate(predicted_next_observations_i):
                for col, bit in enumerate(obs):
                    rec = pygame.Rect(
                        (row + 1) * BIT_SIZE + BIT_SEPARATOR,
                        (self.n_bits + 1 + col) * BIT_SIZE + BIT_SEPARATOR,
                        BIT_SIZE-BIT_SEPARATOR,
                        BIT_SIZE-BIT_SEPARATOR
                    )
                    bit_color = bit_color_scheme[bit]
                    pygame.draw.rect(map_, bit_color, rec)


        return [pygame.surfarray.array3d(map_)]



class TrainableBitFlipper(BitFlipper, base.TrainableDenseToDenseEnv):
    """Hanoi tower environment based on Neural Network."""
    def __init__(self, n_bits=None, modeled_env=None, predict_delta=True,
                 done_threshold=0.5, reward_threshold=0.5):

        super().__init__(n_bits=modeled_env.n_bits,
                         reward_for_solved=modeled_env._reward_for_solved,
                         reward_for_invalid_action=
                         modeled_env._reward_for_invalid_action)
        self.observation_space.dtype = np.float32
        self.done_threshold = done_threshold
        self.reward_threshold = reward_threshold
        self._predict_delta = predict_delta
        self._perfect_env = modeled_env

    def transform_predicted_observations(
            self,
            observations,
            predicted_observation
    ):

        if self._predict_delta:
            predicted_observation = observations + predicted_observation
        clipped_predicted_observation = np.clip(
            predicted_observation, self.observation_space.low,
            self.observation_space.high
        )
        return np.around(clipped_predicted_observation)
