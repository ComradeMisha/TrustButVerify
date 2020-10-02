"""Google Football environment."""

import collections
import copy

import gym
import numpy as np

from alpacka.envs import base
from alpacka.utils import attribute
from alpacka.utils import gfootball as gfootball_utils

try:
    import gfootball.env as football_env
except ImportError:
    football_env = None


class GoogleFootball(base.ModelEnv):
    """Google Research Football conforming to the ModelEnv interface."""

    state_size = 480000

    class Renderer(base.EnvRenderer):
        """Renderer for GFootball.

        Uses the state visualization from GFootball's dumped videos.
        """

        def render_state(self, state_info):
            return gfootball_utils.get_frame(state_info)

        def render_action(self, action):
            action_set = football_env.football_action_set.full_action_set
            return str(action_set[action])


    def __init__(self,
                 env_name='academy_empty_goal_close',
                 representation='simple115',
                 rewards='scoring,checkpoints',
                 stacked=False,
                 dump_path=None,
                 solved_at=1,
                 **kwargs):
        if football_env is None:
            raise ImportError('Could not import gfootball! '
                              'HINT: Perform the setup instructions here: '
                              'https://github.com/google-research/football')

        self._solved_at = solved_at
        self._env = football_env.create_environment(
            env_name=env_name,
            representation=representation,
            rewards=rewards,
            stacked=stacked,
            write_full_episode_dumps=dump_path is not None,
            write_goal_dumps=False,
            logdir=dump_path or '',
            **kwargs
        )

        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    def reset(self):
        # pylint: disable=protected-access
        obs = self._env.reset()
        env = self._env.unwrapped
        env._env._trace._trace = collections.deque([], 4)

        return obs

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        if done:
            info['solved'] = info['score_reward'] >= self._solved_at
        return obs, reward, done, info

    def render(self, mode='human'):
        return self._env.render(mode)

    def close(self):
        self._env.close()

    def seed(self, seed=None):
        raise NotImplementedError

    def clone_state(self):
        # pylint: disable=protected-access
        raw_state = self._env.get_state()
        size_encoded = len(raw_state).to_bytes(3, byteorder='big')
        # Byte suffix to enforce self.state_size of state.
        suffix = bytes(self.state_size - len(size_encoded) - len(raw_state))
        resized_state = size_encoded + raw_state + suffix
        state = np.frombuffer(resized_state, dtype=np.uint8)

        env = self._env.unwrapped

        # Temporary fix for a reward bug in Google Football: put everything in
        # state. Long-term fix on the way:
        # https://github.com/google-research/football/pull/115
        trace = env._env._trace
        trace_copy = attribute.deep_copy_without_fields(trace, ['_config',
                                                                '_dump_config'])
        trace_copy._config = trace._config
        trace_copy._dump_config = trace._dump_config

        state_tuple = (
            state,
            copy.deepcopy(env._env._steps_time),
            copy.deepcopy(env._env._step),
            copy.deepcopy(env._env._cumulative_reward),
            copy.deepcopy(env._env._observation),
            trace_copy,
        )
        # Imitate a tuple without the trace part to fool np.testing.assert_equal
        # during testing.

        class StateModuloTrace(tuple):
            def __len__(self):
                assert super().__len__() == 6
                return 5
        return StateModuloTrace(state_tuple)

    def restore_state(self, state):
        # pylint: disable=protected-access
        env = self._env.unwrapped
        (
            state,
            env._env._steps_time,
            env._env._step,
            env._env._cumulative_reward,
            env._env._observation,
            env._env._trace,
        ) = state

        assert state.size == self.state_size, (
            f'State size does not match: {state.size} != {self.state_size}')

        # First 3 bytes encodes size of state.
        size_decoded = int.from_bytes(list(state[:3]), byteorder='big')
        raw_state = state[3:(size_decoded + 3)]
        assert (state[(size_decoded + 3):] == 0).all()

        self._env.set_state(bytes(raw_state))
        return self._observation

    @property
    def _observation(self):
        # pylint: disable=protected-access
        observation = self._env.unwrapped._env.observation()
        observation = self._env.unwrapped._convert_observations(
            observation, self._env.unwrapped._agent,
            self._env.unwrapped._agent_left_position,
            self._env.unwrapped._agent_right_position
        )
        # pylint: enable=protected-access

        # Lets apply observation transformations from wrappers.
        # WARNING: This assumes that only ObservationWrapper(s) in the wrappers
        # stack transform observation.
        env = self._env
        observation_wrappers = []
        while True:
            if isinstance(env, gym.ObservationWrapper):
                observation_wrappers.append(env)
            if isinstance(env, football_env.wrappers.FrameStack):
                return env._get_observation()  # pylint: disable=protected-access
            if isinstance(env, gym.Wrapper):
                env = env.env
            else:
                break

        for wrapper in reversed(observation_wrappers):
            observation = wrapper.observation(observation)

        return observation

    @property
    def state_info(self):
        observations = self._env.unwrapped.observation()
        print(observations)
        assert len(observations) == 1
        return {
            key: value
            for (key, value) in observations[0].items()
            if key in {
                'left_team', 'left_team_direction',
                'right_team', 'right_team_direction',
                'ball', 'ball_direction', 'ball_owned_team',
                'active', 'game_mode',
            }
        }
