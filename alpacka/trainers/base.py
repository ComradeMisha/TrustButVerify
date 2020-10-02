"""Base class for trainers."""


class Trainer:
    """Base class for trainers.

    Trainer is something that can train a neural network using data from memory.
    In the most basic setup, it just samples data from a replay buffer. By
    abstracting Trainer out, we can also support other setups, e.g. tabular
    learning on a tree.
    """

    def __init__(self, network_signature):
        """No-op constructor just to specify the interface.

        Args:
            network_signature (pytree): Input signature for the network.
        """
        del network_signature

    def add_episode(self, episode, auxiliary_bucket_setting=None):
        """Adds an episode to memory.

        Args:
            episode: (Agent/Trainer-specific) Episode object summarizing the
                collected data for training the TrainableNetwork.
            auxiliary_bucket_setting: (Optional[Dict[str, bool]]) Manually
                passed bucket tags, in case user want to provide them when
                adding episodes to the buffer (in the Runner). They do not
                override episodes' attributes.
        """
        raise NotImplementedError

    def train_epoch(self, network):
        """Runs one epoch of training.

        Args:
            network (TrainableNetwork): TrainableNetwork instance to be trained.
        """
        raise NotImplementedError
