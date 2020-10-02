"""Alpacka initialization."""

import gin

from alpacka.utils import dask

gin.config._OPERATIVE_CONFIG_LOCK = dask.SerializableLock()  # pylint: disable=protected-access
