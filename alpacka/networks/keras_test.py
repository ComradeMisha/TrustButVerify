"""Tests for alpacka.networks.keras."""

import functools
import math
import os
import tempfile

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from alpacka import data
from alpacka.networks import keras as keras_networks


@pytest.fixture
def keras_mlp():
    return keras_networks.KerasNetwork(
        network_signature=data.NetworkSignature(
            input=data.TensorSignature(shape=(13,)),
            output=data.TensorSignature(shape=(1,)),
        )
    )


@pytest.fixture
def dataset():
    ((x_train, y_train), _) = keras.datasets.boston_housing.load_data()
    return (x_train, y_train)


@pytest.mark.parametrize('model_fn,input_shape,output_shape', [
    (keras_networks.mlp, (15,), (1,)),
    (keras_networks.convnet_mnist, (3, 3, 6), (1,)),
])
def test_model_valid(model_fn, input_shape, output_shape):
    network = keras_networks.KerasNetwork(
        model_fn=model_fn,
        network_signature=data.NetworkSignature(
            input=data.TensorSignature(shape=input_shape),
            output=data.TensorSignature(shape=output_shape),
        ),
    )
    batch_size = 7
    inp = np.zeros((batch_size,) + input_shape)
    out = network.predict(inp)
    assert out.shape == (batch_size,) + output_shape


def test_keras_mlp_train_epoch_on_boston_housing(keras_mlp, dataset):
    # Set up
    (x_train, y_train) = dataset
    x_train = x_train[:16]
    y_train = np.expand_dims(y_train[:16], 1)

    def data_stream():
        for _ in range(3):
            yield (x_train, y_train)

    # Run
    metrics = keras_mlp.train(data_stream, 3, 0)

    # Test
    assert 'loss' in metrics


def test_keras_mlp_predict_batch_on_boston_housing(keras_mlp, dataset):
    # Set up
    (data, _) = dataset
    data_batch = data[:16]

    # Run
    pred_batch = keras_mlp.predict(data_batch)

    # Test
    assert pred_batch.shape == (16, 1)


def test_keras_mlp_modify_weights(keras_mlp):
    # Set up
    new_params = keras_mlp.params
    for p in new_params:
        p *= 2

    # Run
    keras_mlp.params = new_params

    # Test
    for new, mlp in zip(new_params, keras_mlp.params):
        assert np.all(new == mlp)


def test_keras_mlp_save_weights(keras_mlp):
    # Set up, Run and Test
    with tempfile.NamedTemporaryFile() as temp_file:
        assert os.path.getsize(temp_file.name) == 0
        keras_mlp.save(temp_file.name)
        assert os.path.getsize(temp_file.name) > 0


def test_keras_mlp_restore_weights(keras_mlp):
    with tempfile.NamedTemporaryFile() as temp_file:
        # Set up
        orig_params = keras_mlp.params
        keras_mlp.save(temp_file.name)

        new_params = keras_mlp.params
        for p in new_params:
            p *= 2
        keras_mlp.params = new_params

        # Run
        keras_mlp.restore(temp_file.name)

        # Test
        for orig, mlp in zip(orig_params, keras_mlp.params):
            assert np.all(orig == mlp)


def test_chop_loss_binary_simple():
    y_true = tf.constant([1, 0, 1], shape=(3, 1), dtype=tf.dtypes.float32)
    y_pred = tf.constant([0.5, 0.9, 1], shape=(3, 1), dtype=tf.dtypes.float32)
    eps = 1.5

    keras_loss = tf.keras.losses.BinaryCrossentropy
    loss_obj = keras_networks.chop_loss(
        keras_loss, eps
    )

    loss = loss_obj(y_true, y_pred).numpy().item()
    assert 1e-3 < loss < 1 - 1e-3


def test_chop_loss_binary_identity():
    y_true = tf.constant([1, 0, 1], shape=(3, 1), dtype=tf.dtypes.float32)
    y_pred = tf.constant([0.5, 0.9, 1], shape=(3, 1), dtype=tf.dtypes.float32)
    eps = 0

    keras_loss = tf.keras.losses.BinaryCrossentropy
    loss_obj = keras_networks.chop_loss(
        keras_loss, eps
    )

    loss = loss_obj(y_true, y_pred).numpy().item()
    true_loss = keras_loss()(y_true, y_pred).numpy().item()
    assert math.isclose(loss, true_loss, rel_tol=1e-3)


def test_chop_loss_categorical():
    keras_loss = functools.partial(
        tf.keras.losses.CategoricalCrossentropy, from_logits=True
    )
    shape = (64, 10, 10, 7)
    true_classes = tf.random.uniform(
        shape[:-1], minval=0, maxval=shape[-1], dtype=tf.dtypes.int32, seed=0
    )
    y_true = tf.one_hot(
        true_classes, shape[-1], dtype=tf.dtypes.float32, name=None
    )
    y_pred = tf.random.uniform(
        shape, minval=-1, maxval=1, dtype=tf.dtypes.float32, seed=1
    )

    blind_loss = keras_networks.chop_loss(keras_loss, 1e9)
    blind_loss_val = blind_loss(y_true, y_pred)
    assert blind_loss_val.numpy().item() == 0.0

    normal_loss = keras_networks.chop_loss(keras_loss, 0)
    normal_loss_val = normal_loss(y_true, y_pred).numpy().item()
    true_loss_val = keras_loss()(y_true, y_pred).numpy().item()
    assert math.isclose(normal_loss_val, true_loss_val, rel_tol=1e-3)
