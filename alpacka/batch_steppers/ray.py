"""Environment steppers."""

import typing

import functools
import gin
import numpy as np

from alpacka.batch_steppers import core

# WA for: https://github.com/ray-project/ray/issues/5250
# One of later packages (e.g. gym_sokoban.envs) imports numba internally.
# This WA ensures its done before Ray to prevent llvm assertion error.
import numba  # pylint: disable=wrong-import-order
import ray  # pylint: disable=wrong-import-order
del numba


_worker_init_hooks = []


def register_worker_init_hook(hook):
    """Add hook called in at initialization of Ray workers

    Args:
        hook: callable
    """
    _worker_init_hooks.append(hook)


class RayObject(typing.NamedTuple):
    """Keeps value and id of an object in the Ray Object Store."""
    id: typing.Any
    value: typing.Any

    @classmethod
    def from_value(cls, value, weakref=False):
        return cls(ray.put(value, weakref=weakref), value)


class RayBatchStepper(core.BatchStepper):
    """Batch stepper running remotely using Ray.

    Runs predictions and steps environments for all Agents separately in their
    own workers.

    It's highly recommended to pass params to run_episode_batch as a numpy array
    or a collection of numpy arrays. Then each worker can retrieve params with
    zero-copy operation on each node.

    Note: RayBatchStepper does not support legacy requests.
    """

    class Worker:
        """Ray actor used to step agent-environment-network in own process."""

        def __init__(
            self, env_class, agent_class, network_fn, model_class,
            model_network_fn, config, init_hooks,
        ):
            # Limit number of threads used between independent tf.op-s to 1.
            import tensorflow as tf  # pylint: disable=import-outside-toplevel
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.threading.set_intra_op_parallelism_threads(1)

            gin.parse_config(config, skip_unknown=True)

            for hook in init_hooks:
                hook()

            self.env = env_class()
            self.agent = (
                agent_class()
                if model_class is None
                else agent_class(model_class=model_class)
            )

            # Metrics cause some problems with Ray, so we switch them off,
            # as we don't train any networks inside the worker.
            if network_fn:
                network_fn = functools.partial(network_fn, metrics=None)
            if model_network_fn:
                model_network_fn = functools.partial(
                    model_network_fn, metrics=None
                )
            self._request_handler = core.RequestHandler(
                network_fn, model_network_fn=model_network_fn
            )

        def run(self, agent_params, model_params, solve_kwargs):
            """Runs the episode using the given network parameters."""
            episode_cor = self.agent.solve(self.env, **solve_kwargs)
            return self._request_handler.run_coroutine(
                episode_cor, agent_params, model_params=model_params
            )

        @property
        def network(self):
            return self._request_handler.agent_network.network

    def __init__(self, env_class, agent_class, network_fn, n_envs, output_dir,
                 model_class=None, model_network_fn=None):
        super().__init__(
            env_class, agent_class, network_fn, n_envs, output_dir,
            model_class, model_network_fn
        )

        config = RayBatchStepper._get_config(
            env_class, agent_class, network_fn,
            model_class, model_network_fn
        )
        ray_worker_cls = ray.remote(RayBatchStepper.Worker)

        if not ray.is_initialized():
            kwargs = {
                # Size of the Plasma object store, hardcoded to 1GB for now.
                'object_store_memory': int(1e9),
            }
            ray.init(**kwargs)
        self.workers = [
            ray_worker_cls.remote(  # pylint: disable=no-member
                env_class, agent_class, network_fn, model_class,
                model_network_fn, config, _worker_init_hooks
            )
            for _ in range(n_envs)
        ]

        self._agent_params = RayObject(None, None)
        self._model_params = RayObject(None, None)
        self._solve_kwargs = RayObject(None, None)

    def run_episode_batch(
            self, agent_params, model_params=None, **solve_kwargs
    ):
        """Runs a batch of episodes using the given network parameters.

        Args:
            agent_params (list of np.ndarray): List of agent network parameters
                as numpy ndarray-s.
            model_params (list of np.ndarray): List of model network parameters
                as numpy ndarray-s.
            **solve_kwargs (dict): Keyword arguments passed to Agent.solve().

        Returns:
            List of completed episodes (Agent/Trainer-dependent).
        """
        self._agent_params = RayBatchStepper._reconstruct_params_if_needed(
            self._agent_params, agent_params
        )
        self._model_params = RayBatchStepper._reconstruct_params_if_needed(
            self._model_params, model_params
        )
        self._solve_kwargs = RayObject.from_value(solve_kwargs)

        episodes = ray.get([
            w.run.remote(
                self._agent_params.id, self._model_params.id,
                self._solve_kwargs.id
            )
            for w in self.workers
        ])
        return episodes

    @staticmethod
    def _reconstruct_params_if_needed(old_params_ray, new_params_np):
        # Optimization, don't send the same parameters again.
        if (old_params_ray.value is None or
                not all([
                    np.array_equal(p1, p2)
                    for p1, p2 in zip(new_params_np, old_params_ray.value)
                ]) or
                isinstance(new_params_np, dict)):
            return RayObject.from_value(new_params_np)
        return old_params_ray

    @staticmethod
    def _get_config(env_class, agent_class, network_fn, model_class,
                    model_network_fn):
        """Returns gin operative config for (at least) env, agent and network.

        It creates env, agent and network to initialize operative gin-config.
        It deletes them afterwords.
        """

        sample_env = env_class()
        agent = agent_class()
        if hasattr(agent, '_value_acc_class'):
            agent._value_acc_class(0.)  # pylint: disable=protected-access
        if network_fn:
            network_fn()
        if model_class:
            model_class(modeled_env=sample_env)
        if model_network_fn:
            model_network_fn()
        return gin.operative_config_str()
