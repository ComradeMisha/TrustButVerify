"""Network interface implementation using the Keras framework."""

import functools

import gin
import numpy as np
import tensorflow as tf
from tensorflow import keras

from alpacka import data
from alpacka.networks import core


def _make_inputs(input_signature):
    """Initializes keras.Input layers for a given signature.

    Args:
        input_signature (pytree of TensorSignatures): Input signature.

    Returns:
        Pytree of tf.keras.Input layers.
    """
    def init_layer(signature):
        return keras.Input(shape=signature.shape, dtype=signature.dtype)
    return data.nested_map(init_layer, input_signature)


def _make_output_heads(hidden, output_signature, output_activation):
    """Initializes Dense layers for heads.

    Args:
        hidden (tf.Tensor): Output of the last hidden layer.
        output_signature (pytree of TensorSignatures): Output signature.
        output_activation (pytree of activations): Activation of every head. See
            tf.keras.layers.Activation docstring for possible values.

    Returns:
        Pytree of head output tensors.
    """
    def init_head(signature, activation, name):
        assert signature.dtype == np.float32
        (depth,) = signature.shape
        return keras.layers.Dense(
            depth, name=name, activation=activation
        )(hidden)
    names = None
    if isinstance(output_signature, dict):
        names = {
            output_name: output_name for output_name in output_signature.keys()
        }
    return data.nested_zip_with(
        init_head,
        (output_signature, output_activation, names)
    )


@gin.configurable
def mlp(network_signature, hidden_sizes=(32,), activation='relu',
        output_activation=None):
    """Simple multilayer perceptron."""
    inputs = _make_inputs(network_signature.input)

    x = inputs
    for h in hidden_sizes:
        x = keras.layers.Dense(h, activation=activation)(x)

    outputs = _make_output_heads(x, network_signature.output, output_activation)
    return keras.Model(inputs=inputs, outputs=outputs)


@gin.configurable
def convnet_mnist(
    network_signature,
    n_conv_layers=5,
    d_conv=64,
    d_ff=128,
    activation='relu',
    output_activation=None,
    global_average_pooling=False,
):
    """Simple convolutional network."""
    inputs = _make_inputs(network_signature.input)

    x = inputs
    for _ in range(n_conv_layers):
        x = keras.layers.Conv2D(
            d_conv, kernel_size=(3, 3), padding='same', activation=activation
        )(x)
    if global_average_pooling:
        x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(d_ff, activation=activation)(x)

    outputs = _make_output_heads(x, network_signature.output, output_activation)
    return keras.Model(inputs=inputs, outputs=outputs)


@gin.configurable
def fcn_with_reward_function(
    network_signature,
    cnn_channels=64,
    cnn_n_layers=2,
    cnn_kernel_size=(5, 5),
    cnn_strides=(1, 1),
    cnn_final_pool_size=(1, 1),
    cnn_l2=0.,
    output_activation=None,
    batch_norm=False,
    global_average_pooling=False
):
    """Fully-convolutional network for (obs, action) to (obs', reward)
    prediction.
    """
    # Last channel does not match because input has one-hot encoded action.
    assert network_signature.input.shape[:-1] == \
           network_signature.output['next_observation'].shape[:-1]
    assert network_signature.input.shape[-1] > \
           network_signature.output['next_observation'].shape[-1]
    inputs = _make_inputs(network_signature.input)

    x = inputs
    for _ in range(cnn_n_layers):
        x = keras.layers.Conv2D(
            cnn_channels, kernel_size=cnn_kernel_size, strides=cnn_strides,
            padding='same', kernel_regularizer=keras.regularizers.l2(cnn_l2),
            activation='relu'
        )(x)
        if batch_norm:
            x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=cnn_final_pool_size)(x)
    x_obs = x

    if global_average_pooling:
        x = keras.layers.GlobalAveragePooling2D()(x)

    if output_activation is None:
        output_activation = {
            'next_observation': keras.activations.softmax,
            'reward': None,
            'done': keras.activations.sigmoid
        }

    final_layers = (
        (x_obs, 'next_observation'),
        (x, 'reward'),
        (x, 'done')
    )

    outputs = {
        layer_name: keras.layers.Dense(
            network_signature.output[layer_name].shape[-1],
            output_activation[layer_name],
            name=layer_name
        )(layer)
        for layer, layer_name in final_layers
    }

    return keras.models.Model(inputs=inputs, outputs=outputs)


@gin.configurable
def two_step_lr(epoch, init_lr, target_lr, n_init_epochs):
    """Two step learning rate schedule.

    All parameters except epoch should be provided by gin.
    """
    if epoch < n_init_epochs:
        return init_lr
    return target_lr


@gin.configurable
def chop_loss(loss_class, threshold):
    """Sets loss to 0 for samples with small loss."""
    loss = loss_class(reduction=tf.keras.losses.Reduction.NONE)
    return functools.partial(_chop_loss, loss, threshold)


def _chop_loss(
        loss, threshold, y_true, y_pred, sample_weight=None
):
    if sample_weight is not None:
        raise ValueError('Argument sample_weight is not supported')

    loss_per_elt = loss(y_true, y_pred)
    loss_per_elt_rank = y_true.shape.rank - 1  # all losses reduce rank by 1
    loss_per_sample = tf.math.reduce_mean(
        loss_per_elt, axis=range(1, loss_per_elt_rank)
    )

    is_relevant = loss_per_sample >= threshold
    relevant_losses = loss_per_sample * tf.cast(
        is_relevant, loss_per_sample.dtype
    )

    return tf.math.reduce_mean(relevant_losses)


@gin.configurable
class PerfectNextObservation(tf.keras.metrics.Metric):
    """Computes perfect next observation statistic.

    Next observation predicted by the model network is perfect iff it matches
    ground truth observation on every single cell (i.e. there are no errors
    anywhere).
    """

    def __init__(self, name='perfect_next_observations', **kwargs):
        super(PerfectNextObservation, self).__init__(name=name, **kwargs)
        self.perfect_observations = self.add_weight(
            name='perfect_observations', initializer='zeros'
        )
        self.total_observations = self.add_weight(
            name='total_observations', initializer='zeros'
        )

    def update_state(self, y_true, y_pred, sample_weight=None):  # pylint: disable=arguments-differ
        """Accumulates number of perfect next observations.

        Args:
            y_true: The ground truth values, with the same dimensions as
                `y_pred`.
            y_pred: The predicted values. Each element must be in the range
                `[0, 1]`.
            sample_weight: Optional weighting of each example. Defaults to 1.
                Can be a `Tensor` whose rank is either 0, or the same rank as
                `y_true`, and must be broadcastable to `y_true`.
        """
        batch_size = tf.shape(y_true)[0]

        y_pred_sparse = tf.argmax(y_pred, axis=-1)
        y_true_sparse = tf.argmax(y_true, axis=-1)

        values = y_pred_sparse == y_true_sparse
        values = tf.cast(values, tf.float32)

        # Reduce each 2D observation to min value. If observation is perfect
        # then min is 1. We leave batch dim intact.
        perfect_observations_counts = tf.math.reduce_min(values, axis=[1, 2])

        # Apply sample weighting if provided.
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            perfect_observations_counts = tf.multiply(
                perfect_observations_counts, sample_weight
            )

        # Update internal counters
        self.perfect_observations.assign_add(
            tf.reduce_sum(perfect_observations_counts)
        )
        self.total_observations.assign_add(tf.cast(batch_size, tf.float32))

    def result(self):
        return self.perfect_observations / self.total_observations

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.perfect_observations.assign(0.)
        self.total_observations.assign(0.)


@gin.configurable
class WeightsLogger(tf.keras.callbacks.Callback):
    """Logs statistics about weights of a network."""
    def __init__(self, log_every_n_epochs=1):
        super().__init__()
        self._log_every_n_epochs = log_every_n_epochs

    def on_epoch_end(self, epoch, logs=None):
        """Logs mean and max absolute weight for every layer."""
        if epoch % self._log_every_n_epochs != 0:
            return
        if logs is None:
            raise ValueError('Expected dict instead of None')

        stats = {}
        for layer_idx, layer in enumerate(self.model.layers):
            layer_key = f'weights_{layer_idx}_{layer.name[:8]}'
            for weights_idx, weights_var in enumerate(layer.trainable_weights):
                weights = weights_var.read_value().numpy()

                shape_str = str(weights.shape).replace(' ', '')
                weights_key = f'{layer_key}_{weights_idx}_{shape_str}_'

                abs_weights = np.absolute(weights)
                stats[f'{weights_key}_mean_abs'] = abs_weights.mean()
                stats[f'{weights_key}_max_abs'] = abs_weights.max()

        if any(logs.keys() & stats.keys()):
            raise ValueError('Naming conflict of weights with other metrics')
        logs.update(stats)


class TabularLookupNetwork(core.TrainableNetwork):
    """Network that saves observations and looks them up."""
    def __init__(self, network_signature, env, **kwargs):
        del kwargs
        super().__init__(network_signature)
        self.transition_table = {}
        self.env = env

    def predict(self, inputs):
        return self.env.lookup_transitions(self.transition_table, inputs)

    @property
    def params(self):
        """Returns network parameters."""
        return self.transition_table

    @params.setter
    def params(self, new_params):
        """Sets network parameters."""
        self.transition_table = new_params

    def train(self, data_stream, n_steps, epoch, validation_data_stream=None):
        """Performs one epoch of training on data prepared by the Trainer."""

        for batch_inputs, batch_outputs in data_stream():
            for obs_action, next_obs, reward, done in zip(
                batch_inputs,
                batch_outputs['next_observation'],
                batch_outputs['reward'],
                batch_outputs['done'],
            ):
                action_space = [0, 1, 2, 3]
                observation = obs_action[:-len(action_space)]
                action = np.argmax(obs_action[-len(action_space):])
                state = self.env.obs2state(observation)
                next_state = self.env.obs2state(next_obs)
                reward, done = reward.item(), done.item()
                transition = state, action, next_state, reward, done
                self.env.update_transition(self.transition_table, transition)

        return {
            'total_transitions_learned': sum(
                len(transitions_s)
                for transitions_s in self.transition_table.values()
            )
        }


class KerasNetwork(core.TrainableNetwork):
    """TrainableNetwork implementation in Keras.

    Args:
        network_signature (NetworkSignature): Network signature.
        model_fn (callable): Function network_signature -> tf.keras.Model.
        optimizer: See tf.keras.Model.compile docstring for possible values.
        loss: See tf.keras.Model.compile docstring for possible values.
        loss_weights (list or None): Weights assigned to losses, or None if
            there's just one loss.
        weight_decay (float): Weight decay to apply to parameters.
        metrics: See tf.keras.Model.compile docstring for possible values
            (Default: None).
        train_callbacks: List of keras.callbacks.Callback instances. List of
            callbacks to apply during training (Default: None)
        **compile_kwargs: These arguments are passed to tf.keras.Model.compile.
    """

    def __init__(
        self,
        network_signature,
        model_fn=mlp,
        optimizer='adam',
        loss='mean_squared_error',
        loss_weights=None,
        weight_decay=0.0,
        metrics=None,
        train_callbacks=None,
        **compile_kwargs
    ):
        super().__init__(network_signature)
        self._model = model_fn(network_signature)
        self._add_weight_decay(self._model, weight_decay)
        self._model.compile(
            optimizer=optimizer,
            loss=loss,
            loss_weights=loss_weights,
            metrics=metrics or [],
            **compile_kwargs
        )

        self.train_callbacks = train_callbacks or []

    @staticmethod
    def _add_weight_decay(model, weight_decay):
        # Add weight decay in form of an auxiliary loss for every layer,
        # assuming that the weights to be regularized are in the "kernel" field
        # of every layer (true for dense and convolutional layers). This is
        # a bit hacky, but still better than having to add those losses manually
        # in every defined model_fn.
        for layer in model.layers:
            if hasattr(layer, 'kernel'):
                # Keras expects a parameterless function here. We use
                # functools.partial instead of a lambda to workaround Python's
                # late binding in closures.
                layer.add_loss(functools.partial(
                    keras.regularizers.l2(weight_decay), layer.kernel
                ))

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

        def dtypes(tensors):
            return data.nested_map(lambda x: x.dtype, tensors)

        def shapes(tensors):
            return data.nested_map(lambda x: x.shape, tensors)

        dataset = tf.data.Dataset.from_generator(
            generator=data_stream,
            output_types=dtypes((self._model.input, self._model.output)),
            output_shapes=shapes((self._model.input, self._model.output)),
        )

        validation_dataset = None
        if validation_data_stream is not None:
            validation_dataset = tf.data.Dataset.from_generator(
                generator=validation_data_stream,
                output_types=dtypes((self._model.input, self._model.output)),
                output_shapes=shapes((self._model.input, self._model.output)),
            )

        # WA for bug: https://github.com/tensorflow/tensorflow/issues/32912
        history = self._model.fit(
            dataset, initial_epoch=epoch, epochs=(epoch + 1), verbose=0,
            steps_per_epoch=n_steps, callbacks=self.train_callbacks,
            validation_data=validation_dataset
        )
        # history contains epoch-indexed sequences. We run only one epoch, so
        # we take the only element.
        return {name: values[0] for (name, values) in history.history.items()}

    def train_one_step(self, x, y, sample_weights, reset_metrics):
        metrics = self._model.train_on_batch(
            x, y, sample_weight=sample_weights, reset_metrics=reset_metrics
        )
        return dict(zip(self._model.metrics_names, metrics))

    def predict(self, inputs):
        """Returns the prediction for a given input.

        Args:
            inputs: (Agent-dependent) Batch of inputs to run prediction on.

        Returns:
            Agent-dependent: Network predictions.
        """

        return self._model.predict_on_batch(inputs)

    @property
    def params(self):
        """Returns network parameters."""

        return self._model.get_weights()

    @params.setter
    def params(self, new_params):
        """Sets network parameters."""

        self._model.set_weights(new_params)

    def save(self, checkpoint_path):
        """Saves network parameters to a file."""

        self._model.save_weights(checkpoint_path, save_format='h5')

    def restore(self, checkpoint_path):
        """Restores network parameters from a file."""

        self._model.load_weights(checkpoint_path)

    def reset(self):
        for layer in self._model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                w_shape = layer.input_shape[1:] + layer.output_shape[1:]
                b_shape = layer.output_shape[1:]
                layer.set_weights([
                    layer.kernel_initializer(w_shape),
                    layer.bias_initializer(b_shape)
                ])
