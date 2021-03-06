"""Batch steppers."""

import gin

from alpacka.batch_steppers import local
from alpacka.batch_steppers import ray


# Configure agents in this module to ensure they're accessible via the
# alpacka.batch_steppers.* namespace.
def configure_batch_stapper(batch_stepper_class):
    return gin.external_configurable(
        batch_stepper_class, module='alpacka.batch_steppers'
    )


LocalBatchStepper = configure_batch_stapper(local.LocalBatchStepper)  # pylint: disable=invalid-name
RayBatchStepper = configure_batch_stapper(ray.RayBatchStepper)  # pylint: disable=invalid-name

# pylint: disable=wrong-import-position
from alpacka.batch_steppers import auto  # This avoids circular imports.
AutoBatchStepper = configure_batch_stapper(auto.AutoBatchStepper)  # pylint: disable=invalid-name
