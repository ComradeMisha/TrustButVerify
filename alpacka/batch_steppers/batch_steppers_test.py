"""Tests for alpacka.batch_steppers."""

import asyncio
import copy
import functools
import platform
import random

from unittest import mock

import gin
import gym
import numpy as np
import pytest
import ray

from alpacka import agents
from alpacka import batch_steppers
from alpacka import data
from alpacka import envs
from alpacka import networks
from alpacka.agents import RandomAgent
from alpacka.data import Request
from alpacka.data import RequestType


class _TestEnv(gym.Env):

    observation_space = gym.spaces.Discrete(1000)
    action_space = gym.spaces.Discrete(1000)

    def __init__(self, actions, n_steps, observations, rewards):
        super().__init__()
        self._actions = actions
        self._n_steps = n_steps
        self._observations = observations
        self._rewards = rewards
        self._step = 0

    def reset(self):
        return self._observations.pop(0)

    def step(self, action):
        self._actions.append(action)
        self._step += 1
        # Assert that we don't do any steps after "done".
        assert self._step <= self._n_steps
        # End the episode at random times.
        done = random.random() < 0.5 or self._step == self._n_steps
        if not done:
            obs = self._observations.pop(0)
        else:
            # Don't take the last observation from the queue, so all sequences
            # are of the same length.
            obs = self.observation_space.sample()
        reward = self._rewards.pop(0)
        return obs, reward, done, {}


class _TestAgent(agents.OnlineAgent):

    def __init__(
        self,
        observations,
        max_n_requests,
        requests,
        responses,
        actions,
    ):
        super().__init__()
        self._observations = observations
        self._max_n_requests = max_n_requests
        self._requests = requests
        self._responses = responses
        self._actions = actions

    def act(self, observation):
        self._observations.append(observation)
        for _ in range(self._max_n_requests):
            # End the predictions at random times.
            if random.random() < 0.5:
                break
            response = yield Request(
                RequestType.AGENT_PREDICTION, np.array([self._requests.pop(0)])
            )
            self._responses.append(response[0])
        return self._actions.pop(0), {}


class _TestNetwork(networks.DummyNetwork):

    def __init__(self, inputs, outputs, metrics=None):
        tensor_sig = data.TensorSignature(shape=(1,))
        super().__init__(
            network_signature=data.NetworkSignature(
                input=tensor_sig, output=tensor_sig
            ),
            metrics=metrics
        )
        self._inputs = inputs
        self._outputs = outputs

    def predict(self, inputs):
        outputs = []
        for x in inputs:
            if x == 0:
                outputs.append(0)
            else:
                self._inputs.append(x)
                outputs.append(self._outputs.pop(0))
        return np.array(outputs)


def _mock_ray_remote(cls):
    class _NewCls:
        def __init__(self, *args, **kwargs):
            self.orig_obj = cls(*args, **kwargs)

        @classmethod
        def remote(cls, *args, **kwargs):
            """Mock Ray Actor factory method."""
            return cls(*args, **kwargs)

        def __getattr__(self, name):
            """Mock every Ray Actor method."""
            orig_attr = self.orig_obj.__getattribute__(name)
            new_attr = mock.Mock()
            new_attr.remote = mock.Mock(side_effect=orig_attr)
            return new_attr

    return _NewCls


def _mock_ray_put_get(x, *args, **kwargs):
    del args
    del kwargs
    return x


def _mock_ray_init(*args, **kwargs):
    del args
    del kwargs


@mock.patch('ray.remote', _mock_ray_remote)
@mock.patch('ray.get', _mock_ray_put_get)
@mock.patch('ray.put', _mock_ray_put_get)
@mock.patch('ray.init', _mock_ray_init)
@pytest.mark.parametrize('batch_stepper_cls', [
    batch_steppers.LocalBatchStepper,
    batch_steppers.RayBatchStepper
])
@pytest.mark.parametrize('max_n_requests', [0, 1, 4])
def test_batch_steppers_run_episode_batch(max_n_requests,
                                          batch_stepper_cls):
    n_envs = 8
    max_n_steps = 4
    n_total_steps = n_envs * max_n_steps
    n_total_requests = n_total_steps * max_n_requests

    # Generate some random data.
    def sample_seq(n):
        return [np.random.randint(1, 999) for _ in range(n)]

    def setup_seq(n):
        expected = sample_seq(n)
        to_return = copy.copy(expected)
        actual = []
        return expected, to_return, actual
    (expected_rew, rew_to_return, _) = setup_seq(n_total_steps)
    (expected_obs, obs_to_return, actual_obs) = setup_seq(n_total_steps)
    (expected_act, act_to_return, actual_act) = setup_seq(n_total_steps)
    (expected_req, req_to_return, actual_req) = setup_seq(n_total_requests)
    (expected_res, res_to_return, actual_res) = setup_seq(n_total_requests)

    # Connect all pipes together.
    stepper = batch_stepper_cls(
        env_class=functools.partial(
            _TestEnv,
            actions=actual_act,
            n_steps=max_n_steps,
            observations=obs_to_return,
            rewards=rew_to_return,
        ),
        agent_class=functools.partial(
            _TestAgent,
            observations=actual_obs,
            max_n_requests=max_n_requests,
            requests=req_to_return,
            responses=actual_res,
            actions=act_to_return,
        ),
        network_fn=functools.partial(
            _TestNetwork,
            inputs=actual_req,
            outputs=res_to_return,
        ),
        n_envs=n_envs,
        output_dir=None,
    )
    episodes = stepper.run_episode_batch(agent_params=None)
    transition_batch = data.nested_concatenate(
        # pylint: disable=not-an-iterable
        [episode.transition_batch for episode in episodes]
    )

    # Assert that all data got passed around correctly.
    assert len(actual_obs) >= n_envs
    np.testing.assert_array_equal(actual_obs, expected_obs[:len(actual_obs)])
    np.testing.assert_array_equal(actual_req, expected_req[:len(actual_req)])
    np.testing.assert_array_equal(actual_res, expected_res[:len(actual_req)])
    np.testing.assert_array_equal(actual_act, expected_act[:len(actual_obs)])

    # Assert that we collected the correct transitions (order is mixed up).
    assert set(transition_batch.observation.tolist()) == set(actual_obs)
    assert set(transition_batch.action.tolist()) == set(actual_act)
    assert set(transition_batch.reward.tolist()) == set(
        expected_rew[:len(actual_obs)]
    )
    assert transition_batch.done.sum() == n_envs


@mock.patch('ray.remote', _mock_ray_remote)
@mock.patch('ray.get', _mock_ray_put_get)
@mock.patch('ray.put', _mock_ray_put_get)
@mock.patch('ray.init', _mock_ray_init)
@pytest.mark.parametrize('batch_stepper_cls', [
    batch_steppers.LocalBatchStepper,
    batch_steppers.RayBatchStepper
])
def test_batch_steppers_network_request_handling(batch_stepper_cls):
    # Set up
    network_class = networks.DummyNetwork
    network_fn = functools.partial(network_class, network_signature=None)
    xparams = 'params'
    episode = 'yoghurt'
    n_envs = 3

    class TestAgent:
        """Dummy agent class."""
        def solve(self, _):
            """Mock solve method."""
            network_fn, params = yield Request(RequestType.AGENT_NETWORK)
            assert isinstance(network_fn(), network_class)
            assert params == xparams
            return episode

    # Run
    bs = batch_stepper_cls(
        env_class=envs.CartPole,
        agent_class=TestAgent,
        network_fn=network_fn,
        n_envs=n_envs,
        output_dir=None,
    )

    # Test
    episodes = bs.run_episode_batch(xparams)
    assert episodes == [episode] * n_envs


class _TestAgentYieldingNetworkRequest:

    def __init__(self, agent_network_class, xparams, n_requests):
        self._agent_network_class = agent_network_class
        self._xparams = xparams
        self._n_requests = n_requests

    @asyncio.coroutine
    def solve(self, _):
        for _ in range(self._n_requests):
            _, params = yield Request(
                RequestType.AGENT_NETWORK
            )
            assert params == self._xparams
        return RequestType.AGENT_NETWORK, self._xparams


class _TestAgentYieldingPredictRequest(agents.OnlineAgent):

    def __init__(self, n_requests, **kwargs):
        super().__init__(**kwargs)
        self._max_n_requests = n_requests
        self._request_type = None
        self._request_value = None

    @asyncio.coroutine
    def solve(self, env, epoch=None, init_state=None, time_limit=None):
        del env
        del epoch
        del init_state
        del time_limit
        for _ in range(self._max_n_requests):
            # End the predictions at random times.
            if random.random() < 0.5:
                break

            response_value = yield Request(
                self._request_type, self._request_value
            )
            self._check_assertions(response_value)

        return 0, {}

    def _check_assertions(self, response_value):
        raise NotImplementedError


class _TestAgentYieldingAgentPredictRequest(_TestAgentYieldingPredictRequest):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._request_type = RequestType.AGENT_PREDICTION
        self._request_value = np.random.rand(3, 3, 3, 5)

    def _check_assertions(self, response_value):
        assert (response_value == self._request_value).all()


class _TestAgentYieldingModelPredictRequest(_TestAgentYieldingPredictRequest):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._request_type = RequestType.MODEL_PREDICTION
        self._request_value = np.random.rand(4, 6, 2)

    def _check_assertions(self, response_value):
        assert (response_value['next_frame'] == self._request_value).all()
        assert (response_value['reward'] == np.ones((4,))).all()
        assert (response_value['done'] == np.zeros((4,), dtype=bool)).all()


class _TestPredictNetwork(networks.DummyNetwork):
    """Mock class for Network."""

    def __init__(self, request_type):
        tensor_sig = data.TensorSignature(shape=(1,))
        super().__init__(
            network_signature=data.NetworkSignature(
                input=tensor_sig, output=tensor_sig
            )
        )
        self._request_type = request_type

    def predict(self, x):
        """Network prediction."""
        if self._request_type == RequestType.AGENT_PREDICTION:
            return x
        elif self._request_type == RequestType.MODEL_PREDICTION:
            batch_size = x.shape[0]
            return {
                'next_frame': x,
                'reward': np.repeat(1, batch_size, axis=0),
                'done': np.repeat(False, batch_size, axis=0)
            }
        else:
            assert False, 'Unexpected request type.'


@pytest.mark.parametrize('batch_stepper_cls', [
    batch_steppers.LocalBatchStepper,
    # batch_steppers.RayBatchStepper
])
def test_batch_steppers_different_requests_single_batch(batch_stepper_cls):
    xparams = 'params'
    n_envs = 9
    n_requests = 100

    agent_network_fn = functools.partial(
        _TestPredictNetwork, request_type=RequestType.AGENT_PREDICTION)
    model_network_fn = functools.partial(
        _TestPredictNetwork, request_type=RequestType.MODEL_PREDICTION)

    bs = batch_stepper_cls(
        env_class=envs.CartPole,
        agent_class=RandomAgent,  # This is only placeholder.
        network_fn=agent_network_fn,  # This is only placeholder.
        model_network_fn=model_network_fn,  # This is only placeholder.
        n_envs=n_envs,
        output_dir=None,
    )

    envs_and_agents = [
        (
            envs.CartPole(),
            _TestAgentYieldingNetworkRequest(
                agent_network_fn, xparams, n_requests
            )
        )
        for _ in range(n_envs // 3)
    ]
    envs_and_agents += [
        (envs.CartPole(), _TestAgentYieldingAgentPredictRequest(n_requests))
        for _ in range(n_envs // 3)
    ]
    envs_and_agents += [
        (envs.CartPole(), _TestAgentYieldingModelPredictRequest(n_requests))
        for _ in range(n_envs // 3)
    ]

    # Envs and agents have to be replaced here manually, because
    # LocalBatchStepper does not support different types of agents to be run
    # simultaneously.
    bs._envs_and_agents = envs_and_agents  # pylint: disable=protected-access
    bs.run_episode_batch(xparams)


class _TestWorker(batch_steppers.RayBatchStepper.Worker):
    def get_state(self):
        return self.env, self.agent, self.network


@mock.patch('alpacka.batch_steppers.ray.RayBatchStepper.Worker', _TestWorker)
@pytest.mark.skipif(platform.system() == 'Darwin')
def test_ray_batch_stepper_worker_members_initialization_with_gin_config():
    # Set up
    solved_at = 7
    env_class = envs.CartPole
    agent_class = agents.RandomAgent
    network_class = networks.DummyNetwork
    n_envs = 3

    gin.bind_parameter('CartPole.solved_at', solved_at)

    env = env_class()
    env.reset()
    root_state = env.clone_state()

    # Run
    bs = batch_steppers.RayBatchStepper(
        env_class=env_class,
        agent_class=agent_class,
        network_fn=functools.partial(network_class, network_signature=None),
        n_envs=n_envs,
        output_dir=None,
    )
    bs.run_episode_batch(None,
                         init_state=root_state,
                         time_limit=10)

    # Test
    assert env.solved_at == solved_at
    assert len(bs.workers) == n_envs
    for worker in bs.workers:
        env, agent, network = ray.get(worker.get_state.remote())
        assert isinstance(env, env_class)
        assert isinstance(agent, agent_class)
        assert isinstance(network, network_class)
        assert env.solved_at == solved_at
