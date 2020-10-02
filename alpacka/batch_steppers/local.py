"""Environment steppers."""

import numpy as np

from alpacka import data
from alpacka.batch_steppers import core
from alpacka.data import Request
from alpacka.data import RequestType


class _NetworkRequestBatcher:
    """Batches network requests.

    There is only one type of network request, which implies that all requests
    are alike.
    """

    def __init__(self, requests):
        if len(requests) == 0:
            raise TypeError(
                'RequestBatcher should be used with at least one Request.'
            )
        self._requests = requests
        self._request = None

    @property
    def batched_request(self):
        """Determines model request and returns it."""
        if self._request is None:
            self._request = self._requests[0]
        return self._request

    def unbatch_responses(self, response):
        return (response for _ in self._requests)


class _PredictionRequestBatcher:
    """Batches prediction requests."""

    def __init__(self, requests):
        if len(requests) == 0:
            raise ValueError('At least one request is required to be batched.')
        self._request_type = requests[0].type
        self._requests = requests
        self._n_requests = len(requests)
        self._batched_request = None

    @property
    def batched_request(self):
        """Batches requests and returns batched request."""
        if self._batched_request is not None:
            return self._batched_request

        data.nested_map(
            _PredictionRequestBatcher._assert_not_scalar, self._requests
        )

        # Stack instead of concatenate to ensure that all requests have
        # the same shape.
        batched_request_content = data.nested_stack([
            request.content for request in self._requests
        ])
        # (n_agents, n_requests, ...) -> (n_agents * n_requests, ...)
        batched_request_content = data.nested_map(
            _PredictionRequestBatcher._flatten_first_2_dims,
            batched_request_content
        )
        self._batched_request = Request(
            self._request_type, batched_request_content
        )
        return self._batched_request

    def unbatch_responses(self, response):
        # (n_agents * n_requests, ...) -> (n_agents, n_requests, ...)
        responses = data.nested_map(
            self._unflatten_first_2_dims, response
        )
        return data.nested_unstack(responses)

    @staticmethod
    def _assert_not_scalar(x):
        if not np.array(x.content).shape:
            raise TypeError(
                'All arrays in a PredictRequest must be at least rank 1.'
            )

    @staticmethod
    def _flatten_first_2_dims(x):
        return np.reshape(x, (-1,) + x.shape[2:])

    def _unflatten_first_2_dims(self, preds):
        return np.reshape(
            preds, (self._n_requests, -1) + preds.shape[1:]
        )


class LocalBatchStepper(core.BatchStepper):
    """Batch stepper running locally.

    Runs batched prediction for all Agents at the same time.
    """

    def __init__(self, env_class, agent_class, network_fn, n_envs, output_dir,
                 model_class=None, model_network_fn=None):
        super().__init__(
            env_class, agent_class, network_fn, n_envs, output_dir,
            model_class, model_network_fn
        )

        def make_env_and_agent():
            env = env_class()
            agent = (
                agent_class()
                if model_class is None
                else agent_class(model_class=model_class)
            )
            return env, agent

        self._envs_and_agents = [make_env_and_agent() for _ in range(n_envs)]
        self._request_handler = core.RequestHandler(
            network_fn, model_network_fn
        )

    @staticmethod
    def _get_request_batcher(requests):
        """Determines the common type of requests and returns a batcher.

        All requests must be of the same type.
        """
        request = requests[0]
        if request.type == RequestType.AGENT_NETWORK:
            request_batcher = _NetworkRequestBatcher(requests)
        elif request.type == RequestType.AGENT_PREDICTION or \
                request.type == RequestType.MODEL_PREDICTION:
            request_batcher = _PredictionRequestBatcher(requests)
        else:
            raise TypeError(
                f'Unknown request.type: got {request.type}, '
                f'which is not known RequestType.'
            )

        return request_batcher

    @staticmethod
    def _group_reqs_by_type(coroutines, requests):
        """Groups requests and coroutines by request type."""
        cors_and_reqs = {req_type: [] for req_type in RequestType}

        for coroutine, request in zip(coroutines, requests):
            if request is None:
                raise TypeError(f'Coroutine {coroutine} already returned.')
            cors_and_reqs[request.type].append((coroutine, request))

        return {
            request_type: cors_and_reqs_group
            for request_type, cors_and_reqs_group in cors_and_reqs.items()
            if len(cors_and_reqs_group) > 0
        }

    @staticmethod
    def _store_transitions(cor, episodes, i):
        """Stores result of the coroutine in the episodes array."""
        episodes[i] = yield from cor
        # End with an infinite stream of Nones, so we don't have
        # to deal with StopIteration later on.
        while True:
            yield None

    @staticmethod
    def _filter_out_finished_coroutines(cors, reqs):
        filtered_cors_and_reqs = [
            (cor, req) for cor, req in zip(cors, reqs) if req is not None
        ]
        if len(filtered_cors_and_reqs) > 0:
            return list(zip(*filtered_cors_and_reqs))
        else:
            return [], []

    def _batch_coroutines(self, cors):
        """Batches a list of coroutines into one.

        Handles batching multiple requests into single one and passing it to
        corresponding batchers.
        """
        # Store the final episodes in a list.
        episodes = [None] * len(cors)
        cors = [
            self._store_transitions(coroutine, episodes, i)
            for (i, coroutine) in enumerate(cors)
        ]
        reqs = [coroutine.send(None) for coroutine in cors]
        reqs = self._wrap_legacy_requests(reqs)
        cors, reqs = self._filter_out_finished_coroutines(cors, reqs)

        while len(reqs) > 0:
            # Requests are grouped by their type and processed in batches,
            # one request type group at a time.
            grouped_reqs_with_cors = LocalBatchStepper._group_reqs_by_type(
                cors, reqs
            )
            # Requests have to be gathered together after processing.
            processed_cors = []
            processed_reqs = []
            for cors_and_reqs_group in grouped_reqs_with_cors.values():
                # Process requests group of same type.
                cors_group, reqs_group = zip(*cors_and_reqs_group)
                batcher = self._get_request_batcher(reqs_group)
                responses = yield batcher.batched_request
                new_reqs = [
                    cor_handle.send(response)
                    for cor_handle, response in zip(
                        cors_group, batcher.unbatch_responses(responses)
                    )
                ]
                processed_cors += cors_group
                processed_reqs += new_reqs

            processed_reqs = self._wrap_legacy_requests(processed_reqs)
            cors, reqs = self._filter_out_finished_coroutines(
                processed_cors, processed_reqs
            )

        return episodes

    def run_episode_batch(
            self, agent_params, model_params=None, **solve_kwargs
    ):
        episode_cor = self._batch_coroutines([
            agent.solve(env, **solve_kwargs)
            for (env, agent) in self._envs_and_agents
        ])
        return self._request_handler.run_coroutine(
            episode_cor, agent_params, model_params
        )
