"""Simple Hanoi Towers gym.
Based on: https://github.com/RobertTLange/gym-hanoi
"""
from copy import deepcopy

from gym import spaces
import numpy as np

from alpacka.envs import base
from alpacka.envs.hanoi_helpers import HanoiHelper
from alpacka.utils.hashing import HashableNdarray

class Hanoi(base.ModelEnv):
    """Hanoi tower adhere to Open AI gym template"""
    metadata = {'render.modes': ['human']}

    def __init__(self, n_disks=None, reward_for_solved=1.,
                 reward_for_invalid_action=0):
        self.n_disks = n_disks
        self._reward_for_solved = reward_for_solved
        self._reward_for_invalid_action = reward_for_invalid_action
        self.action_space = spaces.Discrete(6)
        self.observation_space = \
            spaces.Box(low=np.array([0] * self.n_disks * 3),
                       high=np.array([1] * self.n_disks * 3),
                       dtype=np.uint8
                       )

        self._current_state = None
        self.goal_state = self.n_disks * (2,)

        self.done = None
        self.action_lookup = {0 : '(0,1) - top disk of peg 0 to top of peg 1',
                              1 : '(0,2) - top disk of peg 0 to top of peg 2',
                              2 : '(1,0) - top disk of peg 1 to top of peg 0',
                              3 : '(1,2) - top disk of peg 1 to top of peg 2',
                              4 : '(2,0) - top disk of peg 2 to top of peg 0',
                              5 : '(2,1) - top disk of peg 2 to top of peg 1'}

        self._action_to_move = {0: (0, 1), 1: (0, 2), 2: (1, 0),
                          3: (1, 2), 4: (2, 0), 5: (2, 1)}

        self.helper = HanoiHelper(self.n_disks)
        self._all_states = self.helper.generate_all_states()
        self._all_transitions = self.helper.generate_all_transitions()
        self._visited_states_in_history = set()

    def step(self, action):
        if self.done:
            raise RuntimeError('Episode has finished. '
                               'Call env.reset() to start a new episode.')

        info = {'invalid_action': False}

        move = self._action_to_move[action]

        if self.move_allowed(move):
            disk_to_move = min(self.disks_on_peg(move[0]))
            moved_state = list(self._current_state)
            moved_state[disk_to_move] = move[1]
            self._current_state = tuple(moved_state)
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

    def clone_state(self):
        return HashableNdarray(self.vectorized_obs())

    def restore_state(self, state):
        self._current_state = self.obs2tuple(state.array)
        self.done = self._current_state == self.goal_state

    def obs2tuple(self, obs):
        return tuple(np.reshape(obs, (self.n_disks, 3)).argmax(axis=1))

    @staticmethod
    def state2obs(state):
        return state.array

    @staticmethod
    def obs2state(observation, copy=True):
        if copy:
            observation = deepcopy(observation)
        return HashableNdarray(observation)


    def vectorized_obs(self):
        return np.eye(3)[np.array(self._current_state)].flatten()

    def disks_on_peg(self, peg):
        """
        * Inputs:
            - peg: pole to check how many/which disks are in it
        * Outputs:
            - list of disk numbers that are allocated on pole
        """
        return [disk for disk in range(self.n_disks) if
                self._current_state[disk] == peg]

    def move_allowed(self, move):
        """
        * Inputs:
            - move: tuple of state transition (see ACTION_LOOKUP)
        * Outputs:
            - boolean indicating whether action is allowed from state!
        move[0] - peg from which we want to move disc
        move[1] - peg we want to move disc to
        Allowed if:
            * discs_to is empty (no disc of peg) set to true
            * Smallest disc on target pole larger than smallest on prev
        """
        disks_from = self.disks_on_peg(move[0])
        disks_to = self.disks_on_peg(move[1])

        if disks_from:
            return (min(disks_to) > min(disks_from)) if disks_to else True
        else:
            return False

    def reset(self):
        self._current_state = self.n_disks * (0,)
        self.done = False
        return self.vectorized_obs()

    def render(self, mode='human'):
        for peg in range(3):
            print(f'peg {peg}: {self.disks_on_peg(peg)} ', end='')
        print('')

    def compute_visit_freq_table(self, episodes):
        """Computes visit frequency for each state and averages
        over episodes."""
        visited_states = {state: 0 for state in self._all_states}
        visited_sets_episodes = []
        for episode in episodes:
            visited_set = set()
            if episode.solved:
                visited_set.add(self.goal_state)
            states_batch = [self.obs2tuple(obs) for obs in
                            episode.transition_batch.observation]
            for state in states_batch:
                visited_states[state] += 1
                visited_set.add(state)
                self._visited_states_in_history.add(state)
            visited_sets_episodes.append(visited_set)
            if episode.solved:
                visited_states[self.goal_state] += 1

        visited_freq = np.mean([len(x) / 3 ** self.n_disks
                                for x in visited_sets_episodes])
        visited_states = {state: visited_states[state] /
                                 (len(episodes) * 3 ** self.n_disks)
                          for state in self._all_states}
        return visited_states, visited_freq

    def compute_metrics(self, episodes):
        """Computes environment related metrics."""
        metrics = {}
        _, visit_epoch = self.compute_visit_freq_table(episodes)
        metrics['visited_states_in_epoch'] = visit_epoch
        metrics['visited_states_in_history'] = \
            len(self._visited_states_in_history) / 3 ** (self.n_disks)
        return metrics

    def log_visit_heat_map(self, epoch, episodes, log_detailed_heat_map,
                               metric_logging):

        del log_detailed_heat_map
        visited_states, _ = self.compute_visit_freq_table(episodes)
        heat_map = self.helper.render_heatmap(visited_states)
        metric_logging.log_image(
            f'episode_model/visit_heat_map',
            epoch, heat_map
        )


class TrainableHanoi(Hanoi, base.TrainableDenseToDenseEnv):
    """Hanoi tower environment based on Neural Network."""
    def __init__(self, n_disks=None, modeled_env=None, predict_delta=True,
                 done_threshold=0.5, reward_threshold=0.5):

        super().__init__(n_disks=modeled_env.n_disks,
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
