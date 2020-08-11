import tensorflow as tf
from tensorflow.keras.layers import Layer, ZeroPadding2D, BatchNormalization, Conv2D
import numpy as np


class ConvBlock(Layer):
    def __init__(self, filter_shape, downsample=False, activate=True, batch_norm=True):
        if downsample:
            self._downsample = ZeroPadding2D(((1, 0), (0, 1)))
            padding = 'valid'
            strides = 2
        else:
            strides = 1
            padding = 'same'

        self._conv = Conv2D(filters=filter_shape[-1], kernel_size=filter_shape[0], strides=strides, padding=padding,
                            use_bias=not batch_norm, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                            kernel_initializer=tf.random_normal_initializer(
            stddev=0.01),
            bias_initializer=tf.constant_initializer(0.))

        if batch_norm:
            self._batch_norm = BatchNormalization()
        self._activate = activate

    def call(self, x: tf.Tensor) -> tf.Tensor:
        if self._downsample:
            x = self._downsample(x)
        x = self._conv(x)
        if self._batch_norm:
            x = self._batch_norm(x)
        if self._activate:
            x = tf.nn.leaky_relu(x, alpha=0.1)
        return x


class ResBlock(Layer):
    def __init__(self, input_channel: int, filter_conv1: int, filter_conv2: int):
        self._conv1 = ConvBlock(filter_shape=(
            1, 1, input_channel, filter_conv1))
        self._conv2 = ConvBlock(filter_shape=(
            3, 3, filter_conv1, filter_conv2))

    def call(self, x: tf.Tensor) -> tf.Tensor:
        inputs = x
        x = self._conv1(x)
        x = self._conv2(x)
        res_out = x + inputs
        return res_out
