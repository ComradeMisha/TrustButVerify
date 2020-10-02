"""Metric logging."""
from collections import deque

import gin
import numpy as np


class StdoutLogger:
    """Logs to standard output."""

    @staticmethod
    def log_scalar(name, step, value):
        """Logs a scalar to stdout."""
        # Format:
        #      1 | accuracy:                   0.789
        #   1234 | loss:                      12.345
        #   2137 | loss:                      1.0e-5
        if 0 < value < 1e-2:
            print('{:>6} | {:64}{:>9.1e}'.format(step, name + ':', value))
        else:
            print('{:>6} | {:64}{:>9.3f}'.format(step, name + ':', value))

    @staticmethod
    def log_property(name, value):
        # Not supported in this logger.
        pass

    @staticmethod
    def log_image(name, step, value):
        # Not supported in this logger.
        pass


_loggers = [StdoutLogger]


def register_logger(logger):
    """Adds a logger to log to."""
    _loggers.append(logger)


def log_scalar(name, step, value):
    """Logs a scalar to the loggers."""
    for logger in _loggers:
        logger.log_scalar(name, step, value)


def log_property(name, value):
    """Logs a property to the loggers."""
    for logger in _loggers:
        logger.log_property(name, value)


def log_image(name, step, value):
    """Logs an image to the loggers."""
    for logger in _loggers:
        logger.log_image(name, step, value)


def log_scalar_metrics(prefix, step, metrics):
    for (name, value) in metrics.items():
        log_scalar(prefix + '/' + name, step, value)


def compute_scalar_statistics(x, prefix=None, with_min_and_max=False):
    """Get mean/std and optional min/max of scalar x across MPI processes.

    Args:
        x (np.ndarray): Samples of the scalar to produce statistics for.
        prefix (str): Prefix to put before a statistic name, separated with
            an underscore.
        with_min_and_max (bool): If true, return min and max of x in
            addition to mean and std.

    Returns:
        Dictionary with statistic names as keys (can be prefixed, see the prefix
        argument) and statistic values.
    """
    prefix = prefix + '_' if prefix else ''
    stats = {}

    stats[prefix + 'mean'] = np.mean(x)
    stats[prefix + 'std'] = np.std(x)
    if with_min_and_max:
        stats[prefix + 'min'] = np.min(x)
        stats[prefix + 'max'] = np.max(x)

    return stats


class ExperimentMetric:
    """Calculates metric value across epochs."""

    def update_state(self, epoch, episodes):
        raise NotImplementedError

    def result(self, epoch):
        raise NotImplementedError


@gin.configurable
class EventFirstOccurrenceMetric(ExperimentMetric):
    """Logs first occurrence of an event."""
    event_never_occurred_val = -1

    def __init__(self, check_event_fn, name):
        self._name = name
        self._first_occurrence_epoch = None
        self._check_event_fn = check_event_fn

    def __str__(self):
        return self._name

    def update_state(self, epoch, episodes):
        if (self._first_occurrence_epoch is None and
                self._check_event_fn(episodes)):
            self._first_occurrence_epoch = epoch

    def result(self, epoch):
        return self._first_occurrence_epoch or self.event_never_occurred_val


@gin.configurable
class EventFirstStableOccurrenceMetric(ExperimentMetric):
    """Logs first epoch in which an event occurs repeatedly."""
    event_never_occurred_val = -1

    def __init__(self, check_event_fn, n_epochs, stability_ratio, name):
        self._name = name
        self._first_occurrence_epoch = None
        self._check_event_fn = check_event_fn

        self._last_n_epochs_events = deque(
            np.full((n_epochs,), False), maxlen=n_epochs
        )
        self._stability_ratio = stability_ratio

    def __str__(self):
        return self._name

    def update_state(self, epoch, episodes):
        if self._first_occurrence_epoch is not None:
            return

        self._last_n_epochs_events.append(self._check_event_fn(episodes))
        if np.mean(self._last_n_epochs_events) > self._stability_ratio:
            self._first_occurrence_epoch = epoch

    def result(self, epoch):
        return self._first_occurrence_epoch or self.event_never_occurred_val


@gin.configurable
class StabilityAfterFirstOccurrenceMetric(ExperimentMetric):
    """Calculates ratio of event occurrences after first stable occurrence."""
    default_stability_val = 0

    def __init__(self, check_event_fn, n_epochs, stability_ratio, name):
        self._name = name
        self._first_occurrence_epoch = None
        self._check_event_fn = check_event_fn

        self._last_n_epochs_events = deque(
            np.full((n_epochs,), False), maxlen=n_epochs
        )
        self._stability_ratio = stability_ratio
        self._n_events_after_first_occurrence = 0

    def __str__(self):
        return self._name

    def update_state(self, epoch, episodes):
        if self._first_occurrence_epoch is None:
            self._last_n_epochs_events.append(self._check_event_fn(episodes))
            if np.mean(self._last_n_epochs_events) > self._stability_ratio:
                self._first_occurrence_epoch = epoch
                self._n_events_after_first_occurrence += 1
        elif self._check_event_fn(episodes):
            self._n_events_after_first_occurrence += 1

    def result(self, epoch):
        if self._first_occurrence_epoch is None:
            return self.default_stability_val
        else:
            return (self._n_events_after_first_occurrence /
                    (1 + epoch - self._first_occurrence_epoch))


@gin.configurable
def solved_rate_over_threshold(episodes, threshold):
    solved_rate = sum(
        int(episode.solved) for episode in episodes
        if episode.solved is not None
    ) / len(episodes)

    return solved_rate > threshold
