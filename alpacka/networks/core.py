"""Deep learning framework-agnostic interface for neural networks."""

import numpy as np

from alpacka import data


class Network:
    """Base class for networks."""

    def __init__(self, network_signature, metrics=None):
        """Initializes Network.

        Args:
            network_signature: (NetworkSignature) Network signature.
            metrics: (network-dependent) Training metrics.
        """
        self._network_signature = network_signature
        del metrics

    def predict(self, inputs):
        """Returns the prediction for a given input.

        Args:
            inputs (Agent-dependent): Batch of inputs to run prediction on.
        """
        raise NotImplementedError

    @property
    def params(self):
        """Returns network parameters."""
        raise NotImplementedError

    @params.setter
    def params(self, new_params):
        """Sets network parameters."""
        raise NotImplementedError

    def save(self, checkpoint_path):
        """Saves network parameters to a file."""
        raise NotImplementedError

    def restore(self, checkpoint_path):
        """Restores network parameters from a file."""
        raise NotImplementedError


class TrainableNetwork(Network):
    """Base class for networks that can be trained."""

    def train(self, data_stream, n_steps, epoch, validation_data_stream=None):
        """Performs one epoch of training on data prepared by the Trainer.

        Args:
            data_stream: (Trainer-dependent) Python generator of batches to run
                the updates on.
            n_steps: (int) Number of training steps in the epoch.
            epoch: (int) Current epoch number.
            validation_data_stream: (Trainer-dependent) Python generator of
                batches to run the validation on.

        Returns:
            dict: Collected metrics, indexed by name.
        """
        raise NotImplementedError


class EnsembleNetwork(TrainableNetwork):
    """Wrapper class for using multiple instances of a network."""
    def __init__(
            self, network_signature, network_fn, n_networks,
            *args, **kwargs
    ):
        super().__init__(network_signature)
        self._networks = [
            network_fn(network_signature, *args, **kwargs)
            for _ in range(n_networks)
        ]

    def train(self, data_stream, n_steps, epoch, validation_data_stream=None):
        metrics = [
            network.train(
                data_stream, n_steps, epoch,
                validation_data_stream=validation_data_stream
            )
            for network in self._networks
        ]
        return metrics[0]

    def predict(self, inputs):
        result = [network.predict(inputs) for network in self._networks]
        stacked_result = data.nested_stack(result, axis=-1)
        return stacked_result


    @property
    def params(self):
        return np.stack([network.params for network in self._networks])

    @params.setter
    def params(self, new_params):
        if new_params.shape[0] != len(self._networks):
            raise ValueError(
                'Number of params does not match the number of networks.'
            )
        for i, network in enumerate(self._networks):
            network.params = new_params[i]

    def save(self, checkpoint_path):
        raise ValueError('Not supported')

    def restore(self, checkpoint_path):
        raise ValueError('Not supported')

    def reset(self):
        for network in self._networks:
            network.reset()


class DummyNetwork(TrainableNetwork):
    """Dummy TrainableNetwork for testing."""

    def train(self, data_stream, n_steps, epoch, validation_data_stream=None):
        del data_stream
        return {}

    def predict(self, inputs):
        return inputs

    @property
    def params(self):
        return None

    @params.setter
    def params(self, new_params):
        del new_params

    def save(self, checkpoint_path):
        del checkpoint_path

    def restore(self, checkpoint_path):
        del checkpoint_path
