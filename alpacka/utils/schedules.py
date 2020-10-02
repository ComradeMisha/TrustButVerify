"""Parameter schedules."""


class LinearAnnealing:
    """Implement the linear annealing parameter schedule."""

    def __init__(self, max_value, min_value, n_epochs):
        """Initializes LinearAnnealingSchedule.

        Args:
            max_value (float): Maximal (starting) parameter value.
            min_value (float): Minimal (final) parameter value.
            n_epochs (int): Across how many epochs parameter should reach from
                its starting to its final value.
        """
        self._min_value = min_value
        self._slope = - (max_value - min_value) / (n_epochs - 1)
        self._intersect = max_value

    def __call__(self, epoch):
        return max(self._min_value, self._slope * epoch + self._intersect)
