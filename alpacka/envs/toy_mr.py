"""Toy Montezuma's Revenge environment."""
import asyncio
import enum
import os
import pathlib

import gin
import numpy as np

from alpacka import data
from alpacka.data import Request
from alpacka.data import RequestType
from alpacka.envs import base
from alpacka.envs import toy_mr_backend as backend
from alpacka.envs.base import visualize_transitions
from alpacka.utils.images_logging import visualize_model_predictions


@gin.constants_from_enum
class ToyMRMaps(enum.Enum):
    """Toy Montezuma's Revenge environment."""

    ONE_ROOM = 'one_room_shifted.txt'
    ONE_ROOM_NO_KEY = 'one_room_no_key.txt'
    FOUR_ROOMS = 'four_rooms.txt'
    HALL_WAY = 'hall_way_shifted.txt'
    FULL_MAP = 'full_mr_map.txt'

    ONE_ROOM_20_NO_TRAPS = 'one_room_20x20_no_traps.txt'
    ONE_ROOM_20_SOME_TRAPS = 'one_room_20x20_some_traps.txt'
    ONE_ROOM_20_DENSE_TRAPS = 'one_room_20x20_dense_traps.txt'

    ONE_ROOM_30_NO_TRAPS = 'one_room_30x30_no_traps.txt'
    ONE_ROOM_30_SOME_TRAPS = 'one_room_30x30_some_traps.txt'
    ONE_ROOM_30_DENSE_TRAPS = 'one_room_30x30_dense_traps.txt'

    ONE_ROOM_50_NO_TRAPS = 'one_room_50x50_no_traps.txt'
    ONE_ROOM_50_SOME_TRAPS = 'one_room_50x50_some_traps.txt'
    ONE_ROOM_50_DENSE_TRAPS = 'one_room_50x50_dense_traps.txt'

    NARROW_HALL_WAY_5 = 'narrow_hall_way_5.txt'
    NARROW_HALL_WAY_7 = 'narrow_hall_way_7.txt'
    NARROW_HALL_WAY_11 = 'narrow_hall_way_11.txt'

    NARROW_HALL_WAY_9_TRAPS = 'narrow_hall_way_9_traps.txt'
    NARROW_HALL_WAY_13_TRAPS = 'narrow_hall_way_13_traps.txt'

    TRAPS_SLALOM = 'traps_slalom.txt'


class ToyMR(backend.ToyMR):
    """Toy Montezuma's Revenge environment."""

    MAP_DIR = 'alpacka/envs/mr_maps/'

    def __init__(self, map_file=ToyMRMaps.ONE_ROOM, **kwargs):
        super(ToyMR, self).__init__(
            map_file=os.path.join(ToyMR.MAP_DIR, map_file.value), **kwargs
        )
        self.tabular = False
        self.map_file_enum = map_file
        self.observation_space.dtype = np.float32
        self.visited_rooms_history = set()
        self._perfect_env = self

    def clone_state(self):
        return self.clone_full_state()

    def restore_state(self, state):
        self.restore_full_state(state)
        return self.render(mode='one_hot')

    def restore_state_from_observation(self, observation):
        self.restore_full_state_from_np_array_version(observation)

    def render_observation(self, observation, mode='one_hot'):
        old_state = self.clone_state()
        self.restore_state_from_observation(observation.astype(np.uint8))
        rendered_obs = self.render(mode)
        self.restore_state(old_state)
        return rendered_obs

    def _get_dense_obs_labels(self):
        sample_observation = self.get_state_named_tuple()
        observation_labels = []
        for attr_name, attr_val in sample_observation:
            if len(attr_val) == 1:
                observation_labels.append(attr_name)
            else:
                observation_labels.append(f'{attr_name}_y')
                observation_labels.append(f'{attr_name}_x')

        return observation_labels

    def visualize_transitions(self, transition_batch):
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
        return visualize_transitions(
            transition_batch, self.render_observation,
            actions_labels=('up', 'right', 'down', 'left'),
            observation_labels=self._get_dense_obs_labels()
        )

    def compute_metrics(self, episodes):
        """Computes environment related metrics."""
        nb_visited_rooms = []
        nb_keys_taken = []
        nb_doors_opened = []
        first_visits = {room_loc: [] for room_loc in self.rooms}
        observation_labels = self._get_dense_obs_labels()

        for episode in episodes:
            room_first_visits = data.nested_map(
                lambda idxs: idxs[-1],
                episode.transition_batch.env_info['room_first_visit']
            )
            nb_visited_rooms_e = 0
            for room, first_visit in room_first_visits.items():
                if first_visit is not None:
                    nb_visited_rooms_e += 1
                    first_visits[room].append(first_visit)
                    self.visited_rooms_history.add(room)
            nb_visited_rooms.append(nb_visited_rooms_e)

            last_observation = episode.transition_batch.next_observation[-1]
            keys_taken_mask = [
                1 - last_observation[idx]
                for idx, label in enumerate(observation_labels)
                if 'key_' in label
            ]
            nb_keys_taken_e = sum(keys_taken_mask)
            nb_keys_taken.append(nb_keys_taken_e)

            doors_opened_mask = [
                1 - last_observation[idx]
                for idx, label in enumerate(observation_labels)
                if 'door_' in label
            ]
            nb_doors_opened_e = sum(doors_opened_mask)
            nb_doors_opened.append(nb_doors_opened_e)

        metrics = {
            'mean_visited_rooms_in_episode': np.mean(nb_visited_rooms),
            'total_visited_rooms_in_epoch': sum([
                1 if len(first_visits_room) > 0 else 0
                for first_visits_room in first_visits.values()
            ]),
            'total_visited_rooms_in_history': len(self.visited_rooms_history),
            'mean_keys_taken_in_episode': np.mean(nb_keys_taken),
            'mean_door_opened_in_episode': np.mean(nb_doors_opened),
        }
        for room_loc, visits in first_visits.items():
            metrics[f'visit_freq_{room_loc}'] = len(visits) / len(episodes)
            if len(visits) == 0:
                visits = [-1]
            metrics[f'min_first_visit_{room_loc}'] = min(visits)
            metrics[f'mean_first_visit_{room_loc}'] = np.mean(visits)

        return metrics

    def interesting_transitions(self):
        """Debugging transitions that are exceptionally hard."""
        tmr_map = pathlib.Path(self.map_file).stem

        if tmr_map == 'one_room_shifted':
            return {
                'key_left': (np.array([1, 1, 0, 2, 8, 1, 1, 1, 0]), 3),
                'key_down': (np.array([1, 1, 0, 1, 7, 1, 1, 1, 0]), 2),
                'open_door': (np.array([1, 1, 1, 8, 8, 1, 0, 1, 0]), 2),
                'leave_room': (np.array([1, 1, 0, 8, 9, 0, 0, 1, 0]), 2),
            }
        if tmr_map == 'hall_way_shifted':
            return {
                'key_0_right': (
                    np.array([1, 1, 0, 7, 1, 1, 1, 1, 1, 1, 1, 1, 0]), 1
                ),
                'key_0_up': (
                    np.array([1, 1, 0, 8, 2, 1, 1, 1, 1, 1, 1, 1, 0]), 0
                ),
                'open_door_0': (
                    np.array([1, 1, 1, 8, 4, 1, 1, 1, 0, 1, 1, 1, 0]), 1
                ),
                'leave_room_0': (
                    np.array([1, 1, 0, 9, 4, 0, 1, 1, 0, 1, 1, 1, 0]), 1
                ),
                'key_1_left': (
                    np.array([2, 1, 0, 2, 8, 0, 1, 1, 0, 1, 1, 1, 0]), 3
                ),
                'key_1_down': (
                    np.array([2, 1, 0, 1, 7, 0, 1, 1, 0, 1, 1, 1, 0]), 2
                ),
                'open_door_1': (
                    np.array([2, 1, 1, 8, 4, 0, 1, 1, 0, 0, 1, 1, 0]), 1
                ),
                'leave_room_1': (
                    np.array([2, 1, 0, 9, 4, 0, 0, 1, 0, 0, 1, 1, 0]), 1
                ),
                'key_2_right': (
                    np.array([3, 1, 0, 7, 8, 0, 0, 1, 0, 0, 1, 1, 0]), 1
                ),
                'key_2_down': (
                    np.array([3, 1, 0, 8, 7, 0, 0, 1, 0, 0, 1, 1, 0]), 2
                ),
                'open_door_2': (
                    np.array([3, 1, 1, 8, 4, 0, 0, 1, 0, 0, 0, 1, 0]), 1
                ),
                'leave_room_2': (
                    np.array([3, 1, 0, 9, 4, 0, 0, 0, 0, 0, 0, 1, 0]), 1
                ),
            }
        if tmr_map == 'four_rooms':
            return {
                'room_0_obstacle_0_down': (
                    np.array([1, 1, 0, 5, 2, 1, 1, 1, 0]), 2
                ),
                'room_0_obstacle_0_up': (
                    np.array([1, 1, 0, 6, 4, 1, 1, 1, 0]), 0
                ),
                'room_0_obstacle_0_left': (
                    np.array([1, 1, 0, 8, 3, 1, 1, 1, 0]), 3
                ),
                'room_0_obstacle_0_right': (
                    np.array([1, 1, 0, 1, 3, 1, 1, 1, 0]), 1
                ),
                'room_0_obstacle_1_down': (
                    np.array([1, 1, 0, 7, 5, 1, 1, 1, 0]), 2
                ),
                'room_0_obstacle_1_right': (
                    np.array([1, 1, 0, 5, 6, 1, 1, 1, 0]), 1
                ),
                'room_0_obstacle_1_up': (
                    np.array([1, 1, 0, 7, 7, 1, 1, 1, 0]), 0
                ),
                'room_0_obstacle_2_down': (
                    np.array([1, 1, 0, 7, 7, 1, 1, 1, 0]), 2
                ),
                'room_0_obstacle_2_right': (
                    np.array([1, 1, 0, 5, 8, 1, 1, 1, 0]), 1
                ),
                'room_0_open_door': (
                    np.array([1, 1, 1, 3, 8, 1, 0, 1, 0]), 2
                ),
                'room_0_move_to_room_1': (
                    np.array([1, 1, 0, 9, 7, 1, 1, 1, 0]), 1
                ),
                'room_1_obstacle_0_bottom_right': (
                    np.array([2, 1, 0, 2, 7, 1, 1, 1, 0]), 1
                ),
                'room_1_obstacle_0_top_right': (
                    np.array([2, 1, 0, 2, 2, 1, 1, 1, 0]), 1
                ),
                'room_1_obstacle_0_bottom_left': (
                    np.array([2, 1, 0, 4, 7, 1, 1, 1, 0]), 3
                ),
                'room_1_obstacle_0_top_left': (
                    np.array([2, 1, 0, 4, 2, 1, 1, 1, 0]), 3
                ),
                'room_1_passage_up': (
                    np.array([2, 1, 0, 6, 4, 1, 1, 1, 0]), 0
                ),
                'room_1_passage_down': (
                    np.array([2, 1, 0, 6, 5, 1, 1, 1, 0]), 2
                ),
                'room_1_move_to_room_2': (
                    np.array([2, 1, 0, 8, 9, 1, 1, 1, 0]), 2
                ),
                'room_2_key_right': (
                    np.array([2, 2, 0, 7, 8, 1, 1, 1, 0]), 1
                ),
                'room_2_key_down': (
                    np.array([2, 2, 0, 8, 7, 1, 1, 1, 0]), 2
                ),
                'room_2_move_to_room_1': (
                    np.array([2, 2, 0, 8, 0, 1, 1, 1, 0]), 0
                ),
                'room_2_move_to_room_1_with_key': (
                    np.array([2, 2, 1, 8, 0, 1, 0, 1, 0]), 0
                ),
                'room_1_move_to_room_0_with_key': (
                    np.array([2, 1, 1, 0, 7, 1, 0, 1, 0]), 3
                ),
                'room_0_move_to_room_3': (
                    np.array([1, 1, 0, 3, 9, 0, 0, 1, 0]), 2
                ),
            }

        if tmr_map == 'full_mr_map':
            return {
                'room_0_open_door_left': (
                    np.array(
                        [5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
                    ),
                    3
                ),
                'room_0_leave_room_left': (
                    np.array(
                        [5, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0]
                    ),
                    3
                ),
                'room_0_open_door_right': (
                    np.array(
                        [5, 1, 1, 9, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0]
                    ),
                    1
                ),
                'room_0_leave_room_right': (
                    np.array(
                        [5, 1, 0, 10, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0]
                    ),
                    1
                ),
                'room_0_key_up': (
                    np.array(
                        [5, 1, 0, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
                    ),
                    0
                ),
                'room_0_key_left': (
                    np.array(
                        [5, 1, 0, 2, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
                    ),
                    0
                ),
                'suicide_trap_left': (
                    np.array(
                        [5, 1, 0, 5, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
                    ),
                    3
                ),
                'suicide_trap_right': (
                    np.array(
                        [5, 1, 0, 5, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
                    ),
                    1
                ),
            }

        return {}

    def visualize_hard_transitions(self, predicted_transitions):
        """Generates visualizations of interesting transitions."""
        transitions_visualizations = dict()
        transitions = self.interesting_transitions()
        for name_t, (observation, action) in transitions.items():
            self._perfect_env.restore_state_from_observation(observation)
            next_obs, _, _, _ = self._perfect_env.step(action)
            pred_next_obs = predicted_transitions[name_t]

            transition_observations = [
                self.render_observation(observation, mode='rgb_array'),
                self.render_observation(next_obs, mode='rgb_array'),
                self.render_observation(pred_next_obs, mode='rgb_array'),
            ]
            observations = np.stack(
                (observation, next_obs, pred_next_obs)
            )
            captions_bottom = {
                'observation': ['true', 'predicted']
            }
            captions_top = {
                attr_name: attr_val.tolist()
                for attr_name, attr_val
                in zip(self._get_dense_obs_labels(), observations.T)
            }
            obs_rgb_grid = visualize_model_predictions(
                transition_observations, captions_bottom, captions_top
            )
            transitions_visualizations[name_t] = obs_rgb_grid

        return transitions_visualizations

    def log_visit_heat_map(self, epoch, episodes, log_detailed_heat_map,
                           metric_logging):
        """Handles logging visualization of the heat map"""

        state_visit_freq = {}
        invalid_transitions = {}
        for episode in episodes:
            for observation in episode.transition_batch.observation:
                state = self.obs2state(observation)
                if state not in state_visit_freq:
                    state_visit_freq[state] = 0
                state_visit_freq[state] += 1

            for obs, action, next_obs, next_obs_preds in zip(
                    episode.transition_batch.observation,
                    episode.transition_batch.action,
                    episode.transition_batch.next_observation,
                    episode.transition_batch.agent_info[
                        'children_observations'],
            ):
                next_obs_pred = next_obs_preds[action]
                if not (next_obs == next_obs_pred).all():
                    state = self.obs2state(obs)
                    next_state_pred = self.obs2state(next_obs_pred)
                    if state not in invalid_transitions:
                        invalid_transitions[state] = {}
                    invalid_transitions[state][action] = next_state_pred

        visit_heat_map = self.render_visit_heat_map(
            state_visit_freq, invalid_transitions, separate_by_keys=False
        )
        metric_logging.log_image(
            f'episode_model/visit_heat_map',
            epoch, visit_heat_map
        )
        if log_detailed_heat_map:
            visit_heat_map = self.render_visit_heat_map(
                state_visit_freq, invalid_transitions, separate_by_keys=True
            )
            metric_logging.log_image(
                f'episode_model/visit_heat_map_detailed',
                epoch, visit_heat_map
            )

        novel_transitions = {}
        for episode in episodes:
            if 'novelty_based_action' in episode.transition_batch.agent_info:
                for obs, action, next_obs, is_novel in zip(
                        episode.transition_batch.observation,
                        episode.transition_batch.action,
                        episode.transition_batch.next_observation,
                        episode.transition_batch.agent_info[
                            'novelty_based_action'],
                ):
                    if is_novel:
                        state = self.obs2state(obs)
                        next_state = self.obs2state(next_obs)
                        if state not in novel_transitions:
                            novel_transitions[state] = {}
                        novel_transitions[state][action] = next_state

        visit_heat_map = self.render_visit_heat_map(
            state_visit_freq, novel_transitions, separate_by_keys=False
        )
        metric_logging.log_image(
            f'episode_model/visit_heat_map_novelty_actions',
            epoch, visit_heat_map
        )
        if log_detailed_heat_map:
            visit_heat_map = self.render_visit_heat_map(
                state_visit_freq, novel_transitions, separate_by_keys=True
            )
            metric_logging.log_image(
                f'episode_model/visit_heat_map_novelty_actions_detailed',
                epoch, visit_heat_map
            )


class TrainableToyMR(ToyMR, base.TrainableDenseToDenseEnv):
    """Toy Montezuma's Revenge environment based on Neural Network."""
    def __init__(self, modeled_env=None, predict_delta=True, done_threshold=0.5,
                 reward_threshold=0.5):
        super(TrainableToyMR, self).__init__(
            modeled_env.map_file_enum, max_lives=modeled_env.max_lives,
            absolute_coordinates=modeled_env.absolute_coordinates,
            doors_keys_scale=modeled_env.doors_keys_scale,
            save_enter_cell=modeled_env.save_enter_cell,
            trap_reward=modeled_env.trap_reward
        )
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

    def compute_metrics(self, episodes):
        metrics = super(TrainableToyMR, self).compute_metrics(episodes)
        transitions = self.interesting_transitions()
        for name_t, (observation, action) in transitions.items():
            self._perfect_env.restore_state_from_observation(observation)
            next_obs, _, _, _ = self._perfect_env.step(action)
            pred_next_obs = episodes[0].trainable_env_info[name_t]
            metrics[f'correct_pred_{name_t}'] = np.equal(
                pred_next_obs, next_obs
            ).all().astype(int)
        transitions_experienced_count = {name: 0 for name in transitions}
        for episode in episodes:
            for name_t, (observation_t, action_t) in transitions.items():
                for observation, action in zip(
                    episode.transition_batch.observation,
                    episode.transition_batch.action,
                ):
                    if (np.equal(observation, observation_t).all() and
                            action == action_t):
                        transitions_experienced_count[name_t] += 1
                        break

        for name_t, count in transitions_experienced_count.items():
            metrics[f'freq_{name_t}'] = count / len(episodes)
        return metrics

    def visualize_replay_buffer(self, batch, priorities):
        """Visualizations of transitions with highest priority in the buffer."""
        transitions_visualizations = []
        actions_labels = ('up', 'right', 'down', 'left')
        for (x_input, y_target, y_pred), priority in zip(batch, priorities):
            obs = x_input[:-self.action_space.n]
            action = np.argmax(x_input[-self.action_space.n:]).item()
            actions = np.array((actions_labels[action], actions_labels[action]))
            next_obs = obs + y_target['next_observation']
            next_obs_pred_exact = obs + y_pred['next_observation']
            next_obs_pred_round = self.transform_predicted_observations(
                obs, y_pred['next_observation']
            )

            dones = np.concatenate((y_target['done'], y_pred['done']))
            rewards = np.concatenate((y_target['reward'], y_pred['reward']))
            priorities = np.array((priority, priority))
            loss = ((next_obs - next_obs_pred_exact) ** 2).mean()
            losses = np.array((loss, loss))

            observations_stacked = np.stack(
                (obs, next_obs, next_obs_pred_exact)
            )
            captions_top = {
                attr_name: attr_val.tolist()
                for attr_name, attr_val
                in zip(self._get_dense_obs_labels(), observations_stacked.T)
            }
            captions_bottom = {
                'action': np.atleast_1d(actions).tolist(),
                'reward': np.atleast_1d(rewards).tolist(),
                'done': np.atleast_1d(dones).tolist(),
                'priority': np.atleast_1d(priorities).tolist(),
                'loss': np.atleast_1d(losses).tolist(),
            }
            observations_rgb = [
                self.render_observation(obs, mode='rgb_array'),
                self.render_observation(next_obs, mode='rgb_array'),
                self.render_observation(next_obs_pred_round, mode='rgb_array'),
            ]
            obs_rgb_grid = visualize_model_predictions(
                observations_rgb, captions_bottom, captions_top
            )
            transitions_visualizations.append(obs_rgb_grid)

        return transitions_visualizations


class TabularToyMR(ToyMR, base.TrainableDenseToDenseEnv):
    """Toy Montezuma's Revenge environment based on LookupTable that stores
    seen transitions."""
    def __init__(self, modeled_env=None, **kwargs):
        super().__init__(
            modeled_env.map_file_enum, max_lives=modeled_env.max_lives,
            absolute_coordinates=modeled_env.absolute_coordinates,
            doors_keys_scale=modeled_env.doors_keys_scale,
            save_enter_cell=modeled_env.save_enter_cell,
            trap_reward=modeled_env.trap_reward
        )
        self.observation_space.dtype = np.float32
        self._perfect_env = modeled_env
        self.tabular = True

    @staticmethod
    def update_transitions(transition_table, transitions):
        transitions = np.atleast_2d(transitions)
        for transition in transitions:
            TabularToyMR.update_transition(transition_table, transition)

    @staticmethod
    def update_transition(transition_table, transition):
        state, action, next_state, reward, done = transition

        if state not in transition_table:
            transition_table[state] = {}
        if action not in transition_table[state]:
            next_obs = TabularToyMR.state2obs(next_state)
            transition_table[state][action] = next_obs, reward, done

    @staticmethod
    def _get_default_next_state(observation, action):
        dy, dx = TabularToyMR._get_delta(action)

        agent_pos_y, agent_pos_x = 3, 4

        observation[agent_pos_x] += dx
        observation[agent_pos_y] += dy

        return observation, 0, False

    @staticmethod
    def lookup_transitions(transition_table, transitions):
        """Looks up transition in the transition table."""
        results = []
        for observation_and_action in np.atleast_2d(transitions):
            observation = observation_and_action[:-1]
            action = observation_and_action[-1]

            state = TabularToyMR.obs2state(observation)
            state_transitions = transition_table.get(state, {})
            if action in state_transitions:
                result = state_transitions[action]
            else:
                result = TabularToyMR._get_default_next_state(
                    observation, action
                )
            results.append(result)

        return np.atleast_2d(np.array(results, dtype=object))

    @asyncio.coroutine
    def _batch_predict_steps(self, observations, actions):
        """Predicts next state, reward and done.

        Args:
            observations (np.ndarray): Array of shape (batch, height, width,
                channels) of one-hot encoded observations (along axis=-1).
            actions (np.ndarray): Array of shape (batch,) of actions performed
                by agents.

        Yields:
            request (Request): Model prediction request with one-hot encoded
                input states and actions; handled by RequestHandler.

        Returns:
            next_state (np.ndarray): Array of shape
                (batch, height, width, n_channels) of one-hot encoded state.
            reward (np.ndarray): Array of shape (batch,) of rewards received
                by agents.
            done (np.ndarray): Array of shape (batch,) indicates if episode
                was terminated.
        """
        assert observations.shape[1:] == self.observation_space.shape

        request_content = np.concatenate(
            (observations, np.expand_dims(actions, axis=-1)),
            axis=-1
        )
        res = yield Request(RequestType.MODEL_PREDICTION, request_content)
        next_observations, rewards, dones = res.T
        next_observations = np.stack(next_observations).astype(float)
        rewards = rewards.astype(float)
        dones = dones.astype(bool)
        infos = [{'solved': done} for done in dones]

        return next_observations, rewards, dones, infos

    def compute_metrics(self, episodes):
        metrics = super(TabularToyMR, self).compute_metrics(episodes)
        transitions = self.interesting_transitions()
        transitions_experienced_count = {name: 0 for name in transitions}
        for episode in episodes:
            for name_t, (observation_t, action_t) in transitions.items():
                for observation, action in zip(
                    episode.transition_batch.observation,
                    episode.transition_batch.action,
                ):
                    if (np.equal(observation, observation_t).all() and
                            action == action_t):
                        transitions_experienced_count[name_t] += 1
                        break

        for name_t, count in transitions_experienced_count.items():
            metrics[f'freq_{name_t}'] = count / len(episodes)

        return metrics
