"""Network interface and its implementations."""

import gin
import gin.tf.external_configurables  # TensorFlow external configurables.
import tensorflow as tf

from alpacka.networks import core
from alpacka.networks import keras
from alpacka.networks import tensorflow


# Additional TF external configurables.
gin.external_configurable(tf.nn.softmax_cross_entropy_with_logits,
                          module='tf.nn',
                          name='softmax_cross_entropy_with_logits')

gin.external_configurable(tf.keras.callbacks.LearningRateScheduler,
                          module='tf.keras.callbacks',
                          name='LearningRateScheduler')

gin.external_configurable(tf.keras.losses.BinaryCrossentropy,
                          module='tf.keras.losses',
                          name='BinaryCrossentropy')

gin.external_configurable(tf.keras.losses.binary_crossentropy,
                          module='tf.keras.losses',
                          name='binary_crossentropy')

gin.external_configurable(tf.keras.losses.CategoricalCrossentropy,
                          module='tf.keras.losses',
                          name='CategoricalCrossentropy')

gin.external_configurable(tf.keras.losses.categorical_crossentropy,
                          module='tf.keras.losses',
                          name='categorical_crossentropy')

gin.external_configurable(tf.keras.metrics.Precision,
                          module='tf.keras.metrics',
                          name='Precision')

gin.external_configurable(tf.keras.metrics.Recall,
                          module='tf.keras.metrics',
                          name='Recall')


# Configure networks in this module to ensure they're accessible via the
# alpacka.networks.* namespace.
def configure_network(network_class):
    return gin.external_configurable(
        network_class, module='alpacka.networks'
    )


Network = core.Network  # pylint: disable=invalid-name
TrainableNetwork = core.TrainableNetwork  # pylint: disable=invalid-name
EnsembleNetwork = configure_network(core.EnsembleNetwork)  # pylint: disable=invalid-name
DummyNetwork = configure_network(core.DummyNetwork)  # pylint: disable=invalid-name
KerasNetwork = configure_network(keras.KerasNetwork)  # pylint: disable=invalid-name
TabularLookupNetwork = configure_network(keras.TabularLookupNetwork)  # pylint: disable=invalid-name
TFMetaGraphNetwork = configure_network(tensorflow.TFMetaGraphNetwork)  # pylint: disable=invalid-name
