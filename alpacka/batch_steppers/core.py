"""Environment steppers."""

import numpy as np

from alpacka.data import NetworkRequest
from alpacka.data import Request
from alpacka.data import RequestType


class BatchStepper:
    """Base class for running a batch of steppers.

    Abstracts out local/remote prediction using a Network.
    """

    def __init__(
        self, env_class, agent_class, network_fn, n_envs, output_dir,
        model_class=None, model_network_fn=None
    ):
        """No-op constructor just for documentation purposes.

        Args:
            env_class (type): Environment class.
            agent_class (type): Agent class.
            network_fn (callable): Function () -> Network. Note: we take this
                instead of an already-initialized Network, because some
                BatchSteppers will send it to remote workers and it makes no
                sense to force Networks to be picklable just for this purpose.
            n_envs (int): Number of parallel environments to run.
            output_dir (str or None): Experiment output dir if the BatchStepper
                is initialized from Runner, None otherwise.
            model_class (type): Model class.
            model_network_fn (callable): Function () -> model Network.
        """
        del env_class
        del agent_class
        del network_fn
        del n_envs
        del output_dir
        del model_class
        del model_network_fn

    def run_episode_batch(
            self, agent_params, model_params=None, **solve_kwargs
    ):  # pylint: disable=missing-param-doc
        """Runs a batch of episodes using the given network parameters.

        Args:
            agent_params (Network-dependent): Agent network parameters.
            model_params (Network-dependent): Model network parameters.
            **solve_kwargs (dict): Keyword arguments passed to Agent.solve().

        Returns:
            List of completed episodes (Agent/Trainer-dependent).
        """
        raise NotImplementedError

    @staticmethod
    def _wrap_legacy_request(request):
        """Wraps single legacy request."""
        if isinstance(request, NetworkRequest):
            return Request(RequestType.AGENT_NETWORK)
        elif isinstance(request, np.ndarray):
            return Request(
                RequestType.AGENT_PREDICTION, content=request
            )
        else:
            return None

    @staticmethod
    def _wrap_legacy_requests(requests):
        """Wraps legacy `NetworkRequest` / raw np.array requests into
        Request interface.
        """
        return [
            request if isinstance(request, Request)
            else BatchStepper._wrap_legacy_request(request)
            for request in requests
        ]


class RequestHandler:
    """Handles requests from the agent coroutine to the network."""

    def __init__(self, agent_network_fn, model_network_fn):
        """Initializes RequestHandler.

        Args:
            agent_network_fn (callable): Function () -> Network.
            model_network_fn (callable): Function () -> Network.
        """
        self.agent_network = Network(agent_network_fn)
        self.model_network = Network(model_network_fn)

    def run_coroutine(self, episode_cor, agent_params, model_params):  # pylint: disable=missing-param-doc
        """Runs an episode coroutine using the given network parameters.

        Args:
            episode_cor (coroutine): Agent.solve coroutine.
            agent_params (Network-dependent): Agent network parameters.
            model_params (Network-dependent): Model network parameters.

        Raises:
            TypeError: if received request instance is not of kind Request, or
            TypeError: if received request type is not valid RequestType.

        Returns:
            List of completed episodes (Agent/Trainer-dependent).
        """
        self.agent_network.should_update_params = True
        self.model_network.should_update_params = True

        try:
            request = episode_cor.send(None)
            while True:
                response = self._handle_request(
                    request, agent_params, model_params
                )
                request = episode_cor.send(response)
        except StopIteration as e:
            return e.value  # episodes

    def _handle_request(self, request, agent_params, model_params):
        if not isinstance(request, Request):
            raise TypeError(
                f'Invalid request type: expected object of '
                f'kind Request, got type: {type(request)}: {request}'
            )
        if request.type == RequestType.AGENT_NETWORK:
            response = self.agent_network.handle_network_request(
                agent_params
            )
        elif request.type == RequestType.AGENT_PREDICTION:
            response = self.agent_network.handle_prediction_request(
                request.content, agent_params
            )
        elif request.type == RequestType.MODEL_PREDICTION:
            response = self.model_network.handle_prediction_request(
                request.content, model_params
            )
        else:
            raise TypeError(
                f'Unknown request.type: got {request.type}, '
                f'which is not known RequestType.'
            )

        return response


class Network:
    """Handles network related operations."""
    def __init__(self, network_fn):
        """Initializes Network used by RequestHandler.

        Args:
            network_fn (callable): Function () -> Network.
        """
        self._network_fn = network_fn
        self._network = None  # Lazy initialize if needed
        self.should_update_params = None

    def handle_network_request(self, params):
        return self._network_fn, params

    def handle_prediction_request(self, x, params):
        return self.get_network(params).predict(x)

    def get_network(self, params=None):
        if self._network is None:
            self._network = self._network_fn()
        if params is not None and self.should_update_params:
            self._network.params = params
            self.should_update_params = False
        return self._network
    network = property(get_network)
