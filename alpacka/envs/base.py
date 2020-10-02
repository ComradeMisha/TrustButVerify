"""Base classes related to environments."""

import asyncio

import gym
import numpy as np
import tensorflow as tf

from alpacka import data
from alpacka.data import Request
from alpacka.data import RequestType
from alpacka.data import nested_unzip
from alpacka.utils import space as space_utils
from alpacka.utils.images_logging import visualize_model_predictions
from alpacka.utils.transformations import insert_action_channels


# ========== Basic building-blocks interfaces ========== #


class CloneRestoreInterface:
    def clone_state(self):
        """Returns the current environment state."""
        raise NotImplementedError

    def restore_state(self, state):
        """Restores environment state, returns the observation."""
        raise NotImplementedError


class RestorableFromObservationInterface:
    """Interface for envs, which can convert observation to state"""

    def restore_state_from_observation(self, observation):
        """Restores environment state based on the given observation.

        It is intended to use, when model mispredicted the current state and we
        want to correct it using the observation given by the true environment.

        Returns: None

        Note:
            Conversion observation->state may be non-trivial for some
            environments (e.g. partially observable environments or
            environments, where the current number of steps matters).

            Thus some environments may not fit into this interface.
        """
        raise NotImplementedError


class GymEnvSubset:
    """Subset of gym.Env methods needed for TrainableModelEnv.

    Classes implementing this interface, should provide also attributes
    described in gym.Env:
        action_space, observation_space and reward_range.
    """
    def render(self, mode='human'):
        """Renders the environment."""
        raise NotImplementedError


# ========== Complete interfaces to be implemented by envs/models ========== #

class ModelEnv(gym.Env, CloneRestoreInterface):
    """Environment interface used by model-based agents.

    This class defines an additional interface over gym.Env that is assumed by
    model-based agents. It's just for documentation purposes, doesn't have to be
    subclassed by envs used as models (but it can be).
    """
    def compute_metrics(self, episodes):
        del episodes
        return {}


class TrainableModelEnv(ModelEnv):
    """Base class for trainable envs that are using Neural Networks."""
    def __init__(self):
        self.observation_space = None
        self.action_space = None
        self.done_threshold = None
        self.reward_threshold = None

    @staticmethod
    def obs2state(observation, copy=True):
        raise NotImplementedError()

    @asyncio.coroutine
    def predict_steps(self, starting_observation, actions):
        """Predicts environment's behavior on every given action.

        This function may use `yield` mechanism to query neural network.

        Please note, that this function (as opposed to gym.Env.step())
        doesn't change model's internal state. If you want to change model's
        state, please use one of methods:
            self.restore_state()
            self.restore_state_from_observation()

        Args:
            starting_observation: Starting observation.
            actions (list(int)): List of actions to try out.

        Yields:
            A stream of Requests for inference in batch stepper.

        Returns:
            A 5-tuple of lists - each of size len(actions):
            (observations, rewards, dones, infos, states)

            States contain internal states of the model
            (like these obtained by ModelEnv.clone_state())
            after performing each action.

            Elements of all other lists is analogous to return values
            of gym.Env.step().
        """
        stacked_starting_observation = np.repeat(
            np.expand_dims(starting_observation.array, axis=0),
            repeats=len(actions),
            axis=0
        )

        pred_obs, rewards, dones, infos = yield from self._batch_predict_steps(
            stacked_starting_observation, np.array(actions)
        )

        pred_obs = pred_obs.astype(self.observation_space.dtype)
        return (
            list(pred_obs),
            list(rewards),
            list(dones),
            list(infos),
            [self.obs2state(obs, copy=False) for obs in pred_obs]
        )

    @asyncio.coroutine
    def predict_step(self, observation, action, repeat_fix=True):
        """Predicts next state, reward and done.

        Args:
            observation (np.ndarray): Array of shape (height, width, n_channels)
                of one-hot encoded observation (along axis=-1).
            action (int): Action performed by the agent.
            repeat_fix (bool): Indicates if 'hack' fix should be used. There is
                a problem with passing requests to RequestHandler that shapes
                of tensors have to be the same for each request, and in this
                case we always pass tensor with stacked predictions requests
                for each action in the state.

        Yields:
            request (Request): Model prediction request with one-hot encoded
                input state and action; handled by RequestHandler.

        Returns:
            pred_obs (np.ndarray): Array of shape (height, width,
                n_channels) of one-hot encoded state (along axis=-1).
            reward (float): Reward received by the agent.
            done (bool): Indicates if episode terminated.
            info (dict): Environment additional info.
        """
        if repeat_fix:
            actions = list(space_utils.element_iter(self.action_space))
            next_obs, rewards, dones, _, _ = yield from self.predict_steps(
                self.obs2state(observation), actions
            )
            return next_obs[action], rewards[action], dones[action]
        batched_observation = np.expand_dims(observation, axis=0)
        batched_action = np.expand_dims(action, axis=0)

        pred_obs, reward, done, info = yield from self._batch_predict_steps(
            batched_observation, batched_action
        )

        # Unbatch predictions
        pred_obs = tf.squeeze(pred_obs, axis=0).numpy()
        reward = reward.item()
        done = done.item()
        info = info.item()

        return pred_obs, reward, done, info

    def transform_predicted_observations(
            self,
            observations,
            predicted_observation
    ):
        raise NotImplementedError()

    def transform_model_outputs(self, observations, model_outputs):
        """Transforms Network predictions to valid environment outputs."""
        next_observations = self.transform_predicted_observations(
            observations, model_outputs['next_observation']
        )

        reward_probas = model_outputs['reward']
        reward_probas = tf.squeeze(reward_probas, axis=-1).numpy()
        rewards = (reward_probas > self.reward_threshold).astype(int)

        done_probas = model_outputs['done']
        done_probas = tf.squeeze(done_probas, axis=-1).numpy()
        dones = done_probas > self.done_threshold

        infos = [{'solved': done} for done in dones]
        return next_observations, rewards, dones, infos

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
        xs = insert_action_channels(observations, actions, self.action_space.n)
        model_outputs = yield Request(RequestType.MODEL_PREDICTION, xs)
        return self.transform_model_outputs(observations, model_outputs)

    @staticmethod
    def wrap_perfect_env(env):
        return _TrainableModelEnvWrapper(env)


class TrainableEnsembleModelEnv(TrainableModelEnv):
    """Wrapper for trainable env that uses Ensembles of Neural Networks."""
    _index_mask = None
    _ensemble_size = 1

    def __init__(self, modeled_env, model_class):
        super().__init__()
        self._env = model_class(modeled_env=modeled_env)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.done_threshold = self._env.done_threshold
        self.reward_threshold = self._env.reward_threshold

    @classmethod
    def set_global_index_mask(cls, ensemble_size, n_ensembles_per_episode):
        """Specifies indices of ensembles used by all accumulators globally."""
        cls._index_mask = np.random.choice(
            ensemble_size, n_ensembles_per_episode, replace=False
        )
        cls._ensemble_size = ensemble_size

    def obs2state(self, observation, copy=True):
        return self._env.obs2state(observation, copy)

    def state2obs(self, state):
        return self._env.state2obs(state)

    def interesting_transitions(self):
        return self._env.interesting_transitions()

    def visualize_transitions(self, transition_batch):
        return self._env.visualize_transitions(transition_batch)

    def visualize_hard_transitions(self, episodes):
        return self._env.visualize_hard_transitions(episodes)

    def compute_metrics(self, episodes):
        return self._env.compute_metrics(episodes)

    def render_visit_heat_map(self, epoch, episodes):
        return self._env.render_visit_heat_map(epoch, episodes)

    def log_visit_heat_map(
        self,
        epoch,
        episodes,
        log_detailed_heat_map,
        metric_logging
    ):
        self._env.log_visit_heat_map(
            epoch, episodes, log_detailed_heat_map, metric_logging)

    def transform_predicted_observations(
            self,
            observations,
            predicted_observation
    ):
        return self._env.transform_predicted_observations(
            observations,
            predicted_observation
        )

    def transform_model_outputs(self, observations, model_outputs):
        def reduce_fn(x):
            return np.mean(x, axis=-1)

        def take_fn(x):
            return np.take(x, self._index_mask, axis=-1)

        masked_model_outputs = data.nested_map(take_fn, model_outputs)
        ensemble_uncertainties = np.std(
            masked_model_outputs['next_observation'],
            axis=-1,
            dtype=np.float64
        )
        bonuses = np.mean(
            ensemble_uncertainties,
            axis=tuple(range(1, ensemble_uncertainties.ndim))
        )
        reduced_model_outputs = data.nested_map(reduce_fn, masked_model_outputs)
        next_observations, rewards, dones, infos = super(
            TrainableEnsembleModelEnv, self
        ).transform_model_outputs(observations, reduced_model_outputs)
        for info, bonus in zip(infos, bonuses):
            info.update({'bonus': bonus})
        return next_observations, rewards, dones, infos


class TrainableFrameToFrameEnv(TrainableModelEnv):
    """Base class for trainable models that predicts frame to frame."""
    @staticmethod
    def obs2state(observation, copy=True):
        raise NotImplementedError()

    def transform_predicted_observations(
            self,
            observations,
            predicted_observation
    ):
        del observations
        board_channels = predicted_observation.shape[-1]
        # Convert softmax predictions to one-hot encoding.
        predicted_observation_dense = tf.argmax(predicted_observation, axis=-1)
        predicted_observation_encoded = tf.one_hot(
            predicted_observation_dense, depth=board_channels
        )
        return predicted_observation_encoded.numpy()


class TrainableDenseToDenseEnv(TrainableModelEnv):
    """Base class for trainable models that operates on dense observations."""
    @staticmethod
    def obs2state(observation, copy=True):
        raise NotImplementedError()

    def _compute_trainable_env_info(self):
        return {}

    def interesting_transitions(self):
        return {}

    def transform_predicted_observations(
            self,
            observations,
            predicted_observation
    ):
        raise NotImplementedError()


class _TrainableModelEnvWrapper:
    """Wrapper class for Envs, presents TrainableModelEnv interface."""

    def __init__(self, env):
        """
        Params:
            env: Environment to wrap.
        """
        self._env = env

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def reward_range(self):
        return self._env.reward_range

    def clone_state(self):
        return self._env.clone_state()

    def render_state(self, state, mode='one_hot'):
        old_state = self._env.clone_state()
        self._env.restore_state(state)
        results = self._env.render(mode)
        self._env.restore_state(old_state)
        return results

    def render_observation(self, observation, mode='one_hot'):
        del observation
        del mode
        raise TypeError('Function render_observation is not supported in '
                         'wrapped envs.')

    def state2obs(self, state):
        old_state = self._env.clone_state()
        self._env.restore_state(state)
        obs = self._env.render()
        self._env.restore_state(old_state)
        return obs

    @asyncio.coroutine
    def predict_steps(self, base_state, actions):
        """Wraps env.step() into common interface for TrainableEnvs."""
        old_state = self._env.clone_state()

        self._env.restore_state(base_state)

        def step_and_rewind(action):
            observation, reward, done, info = self._env.step(action)
            state = self._env.clone_state()
            self._env.restore_state(base_state)
            return observation, reward, done, info, state

        results = list(zip(*[
            step_and_rewind(action)
            for action in actions
        ]))

        self._env.restore_state(old_state)
        return results


class EnvRenderer:
    """Base class for environment renderers."""

    def __init__(self, env):
        """Initializes EnvRenderer."""
        del env

    def render_state(self, state_info):
        """Renders state_info to an image."""
        raise NotImplementedError

    def render_action(self, action):
        """Renders action to a string."""
        raise NotImplementedError


def get_model_network_signature(observation_space, action_space):
    """Defines the signature of the network of the model, used in model-based
    experiments with trainable model.

    Args:
        observation_space (gym.Space): Environment observation space.
        action_space (gym.Discrete): Environment action space.

    Returns:
        NetworkSignature: Signature of the network.
    """
    # Actions are one-hot encoded as entire layers in the input.
    input_channels = observation_space.shape[-1] + action_space.n
    input_shape = *observation_space.shape[:-1], input_channels
    return data.NetworkSignature(
        input=data.TensorSignature(input_shape, dtype=np.float32),
        output={
            'next_observation': data.TensorSignature(
                shape=observation_space.shape, dtype=np.float32
            ),
            'reward': data.TensorSignature(shape=(1,), dtype=np.float32),
            'done': data.TensorSignature(shape=(1,), dtype=np.float32)
        }
    )


def visualize_transitions(transition_batch, render_obs_fn, actions_labels,
                          observation_labels=None):
    """Generates visualizations of agent transitions.

    Single visualization is made of true observation at time step t and
    predicted next observations by the TrainableModel.

    Args:
        transition_batch (data.Transition): Transition object containing
            sequence of transitions to be visualized.
        render_obs_fn (Callable[np.ndarray, np.ndarray]): Function that renders
            observation from one-hot encoding to rgb array.
        actions_labels (List[str]): Human-friendly names for actions.
        observation_labels (Optional[List[str]]): Human-friendly names of
            observation in dense representation, label for each real value in
            the observation vector. If not specified then observation in dense
            representation will not be logged.
    Returns:
        prediction_grids (np.ndarray): Array with RGB images, visualizations
        of TrainableModel predictions at consecutive time steps. It has
        shape of (episode_length, image_H, image_W).
    """
    # Captions over rendered observations on visualization.
    episode_length = len(transition_batch.reward)
    agent_info = transition_batch.agent_info
    captions_info = {
        'action': np.tile(actions_labels, (episode_length, 1)),
        'reward': agent_info.pop('children_rewards'),
        'done': agent_info.pop('children_dones'),
    }

    for info_name, info_val in agent_info.items():
        if info_val.shape == captions_info['action'].shape:
            captions_info[info_name] = info_val

    observations = transition_batch.observation
    children_observations = transition_batch.agent_info[
        'children_observations'
    ]

    # Iterate arrays along time steps `t` in the episode.
    # Subscript t indicates time step.
    prediction_grids = []

    # These awful hacks had to be done because alpacka.data.nested_unzip
    # 1) does not know how to handle numpy.arrays, 2) unzips on axis=-1
    # instead of axis=0. And btw: I thought that if I want to `zip`
    # (built-in fn) nested objects I am supposed to use `nested_zip`, but it
    # turns out `nested_zip` is an inverse of `zip`, so I have to use
    # `nested_unzip`.
    transposed_captions_info = {
        name: np.transpose(arr).tolist() for name, arr in
        captions_info.items()
    }
    for observation_t, children_observations_t, caps_tr in nested_unzip(
            [
                np.transpose(observations).tolist(),
                np.transpose(children_observations).tolist(),
                transposed_captions_info
            ]):
        observation_t = np.array(observation_t).transpose()
        children_observations_t = np.array(
            children_observations_t
        ).transpose()
        captions_bottom = {
            name: np.array(arr).transpose().tolist()
            for name, arr in caps_tr.items()
        }
        # Render observations to RGB.
        observation_rgb_t = render_obs_fn(observation_t, mode='rgb_array')
        children_observations_rgb_t = [
            render_obs_fn(children_obs_t_i, mode='rgb_array')
            for children_obs_t_i in children_observations_t
        ]
        captions_top = None
        if observation_labels is not None:
            observations = np.concatenate(
                (np.expand_dims(observation_t, axis=0), children_observations_t)
            )
            captions_top = {
                attr_name: attr_val.tolist()
                for attr_name, attr_val
                in zip(observation_labels, observations.T)
            }

        # Combine observation and predicted next observations on one plot.
        obs_combined_t = [observation_rgb_t] + children_observations_rgb_t
        obs_rgb_grid = visualize_model_predictions(
            obs_combined_t, captions_bottom, captions_top
        )
        prediction_grids.append(obs_rgb_grid)

    return np.array(prediction_grids)
