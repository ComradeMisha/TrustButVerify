"""Utilities for logging and updating distances in environment graph."""

import time

from alpacka import metric_logging
from alpacka.utils.graph import calc_distance_to_closest_goal
from alpacka.utils.graph import generate_env_state_space_graph_and_goal_states


class GraphDistanceLogger:
    """Calculates state-space graph and updates distances from trajectories."""
    def __init__(self, env):
        assert hasattr(env, 'obs2state')
        time_stamp = time.time()
        self._graph_env = env
        self._graph, goal_states = \
            generate_env_state_space_graph_and_goal_states(self._graph_env)
        self._graph_env.reset()
        start_state = env.clone_state()
        self._distance_from_start = self._graph.calc_distances_from(start_state)
        self._distance_to_goal = calc_distance_to_closest_goal(
            goal_states, self._graph
        )
        self._min_distance_to_goal = float('inf')
        self._max_distance_from_start = 0
        metric_logging.log_scalar(
            'graph/size',
            step=0,
            value=len(
                self._graph.edges.keys() | self._graph.edges_transposed.keys()
            )
        )
        metric_logging.log_scalar(
            'run_time/graph_calculation',
            step=0,
            value=time.time() - time_stamp
        )

    def update_and_log(self, episodes, epoch):
        """

        Args:
            episodes: List of completed episodes (Agent/Trainer-dependent).
            epoch (int): epoch number
        """
        time_stamp = time.time()
        episode_min_goal_distance = []
        episode_max_start_distance = []
        for episode in episodes:
            distances_to_goal = list()
            distances_from_start = list()
            for observation in (
                list(episode.transition_batch.observation) +
                [episode.transition_batch.next_observation[-1]]
            ):
                state = self._graph_env.obs2state(observation)
                distances_to_goal.append(
                    self._distance_to_goal.get(state, float('inf'))
                )
                distances_from_start.append(self._distance_from_start[state])
            episode_min_goal_distance.append(min(distances_to_goal))
            episode_max_start_distance.append(max(distances_from_start))
        self._max_distance_from_start = max(
            self._max_distance_from_start,
            max(episode_max_start_distance)
        )
        self._min_distance_to_goal = min(
            self._min_distance_to_goal,
            min(episode_min_goal_distance)
        )
        self._log_graph_distance_metrics(epoch)
        metric_logging.log_scalar_metrics(
            'run_time',
            epoch,
            {'update_and_log_graph_distances': time.time() - time_stamp}
        )

    def _log_graph_distance_metrics(self, epoch):
        metric_logging.log_scalar_metrics(
            'graph',
            epoch,
            {
                'max_distance_from_start_in_history':
                    self._max_distance_from_start,
                'min_distance_to_goal_in_history':
                    self._min_distance_to_goal
            }
        )
