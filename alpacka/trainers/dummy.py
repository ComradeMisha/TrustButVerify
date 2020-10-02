"""Dummy Trainer for testing."""

import gin

from alpacka.trainers import base


@gin.configurable
class DummyTrainer(base.Trainer):
    """Dummy Trainer for testing and use with plain Networks (not trainable)."""

    def add_episode(self, episode, auxiliary_bucket_setting=None):
        del episode
        del auxiliary_bucket_setting

    def train_epoch(self, network):
        return {}
