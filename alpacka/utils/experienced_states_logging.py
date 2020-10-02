"""Utilities for logging and updating number of experienced states."""

import time

from alpacka import metric_logging


class ExperiencedStatesLogger:
    """Stores visited states and transitions, logs their numbers."""
    def __init__(self, obs2state_fn):
        """ Initialized ExpericedStatesLogger

        Args:
            obs2state_fn: Function mapping observation to hashable state.
        """
        self._experienced_states = set()
        self._experienced_transitions = set()
        self._obs2state = obs2state_fn

    def _get_experienced_states_and_transitions(self, episodes):
        experienced_states = set()
        experienced_transitions = set()
        for episode in episodes:
            for action, observation in zip(
                episode.transition_batch.action,
                episode.transition_batch.observation,
            ):
                state = self._obs2state(observation)
                experienced_states.add(state)
                experienced_transitions.add(
                    (state, action)
                )
            experienced_states.add(self._obs2state(
                episode.transition_batch.next_observation[-1]
            ))
        return experienced_states, experienced_transitions

    def _update(self, episodes):
        epoch_experienced_states, epoch_experienced_transitions = \
            self._get_experienced_states_and_transitions(episodes)
        self._experienced_transitions |= epoch_experienced_transitions
        self._experienced_states |= epoch_experienced_states
        return len(epoch_experienced_states), len(epoch_experienced_transitions)

    def update_and_log(self, episodes, epoch):
        """

        Args:
            episodes: List of completed episodes (Agent/Trainer-dependent).
            epoch (int): epoch number
        """
        time_stamp = time.time()
        n_epoch_states, n_epoch_transitions = self._update(episodes)
        metric_logging.log_scalar_metrics(
            'run_time',
            epoch,
            {'update_experienced_states': time.time() - time_stamp}
        )

        metric_logging.log_scalar_metrics(
            'exploration',
            epoch,
            {
                'total_experienced_states_in_history':
                    len(self._experienced_states),
                'total_experienced_transitions_in_history':
                    len(self._experienced_transitions),
                'total_experienced_states_in_epoch':
                    n_epoch_states,
                'total_experienced_transitions_in_epoch':
                    n_epoch_transitions,
            }
        )
