"""Environments."""

import gin

from alpacka.envs import bit_flipper
from alpacka.envs import cartpole
from alpacka.envs import gfootball
from alpacka.envs import hanoi
from alpacka.envs import sokoban
from alpacka.envs import toy_mr
from alpacka.envs.base import *
from alpacka.envs.wrappers import *


# Configure envs in this module to ensure they're accessible via the
# alpacka.envs.* namespace.
def configure_env(env_class):
    return gin.external_configurable(
        env_class, module='alpacka.envs'
    )



TrainableEnsembleModelEnv = configure_env(TrainableEnsembleModelEnv)  # pylint: disable=invalid-name
BitFlipper = configure_env(bit_flipper.BitFlipper)  # pylint: disable=invalid-name
CartPole = configure_env(cartpole.CartPole)  # pylint: disable=invalid-name
GoogleFootball = configure_env(gfootball.GoogleFootball)  # pylint: disable=invalid-name
Hanoi = configure_env(hanoi.Hanoi)  # pylint: disable=invalid-name
Sokoban = configure_env(sokoban.Sokoban)  # pylint: disable=invalid-name
ToyMR = configure_env(toy_mr.ToyMR)  # pylint: disable=invalid-name
TrainableBitFlipper = configure_env(bit_flipper.TrainableBitFlipper) # pylint: disable=invalid-name
TrainableHanoi = configure_env(hanoi.TrainableHanoi)  # pylint: disable=invalid-name
TrainableSokoban = configure_env(sokoban.TrainableSokoban)  # pylint: disable=invalid-name
TrainableToyMR = configure_env(toy_mr.TrainableToyMR)  # pylint: disable=invalid-name
TabularToyMR = configure_env(toy_mr.TabularToyMR)  # pylint: disable=invalid-name
