"""Test parameter schedules."""

from alpacka import testing
from alpacka.agents import core
from alpacka.utils import schedules

mock_env = testing.mock_env_fixture


def test_linear_annealing_schedule(mock_env):
    # Set up
    attr_name = 'pied_piper'
    param_values = list(range(10, 0, -1))
    max_value = max(param_values)
    min_value = min(param_values)
    n_epochs = len(param_values)

    agent = core.RandomAgent(parameter_schedules={
        attr_name: schedules.LinearAnnealing(max_value, min_value, n_epochs)
    })

    # Run & Test
    for epoch, x_value in enumerate(param_values):
        testing.run_without_suspensions(agent.solve(mock_env, epoch=epoch))
        assert getattr(agent, attr_name) == x_value
