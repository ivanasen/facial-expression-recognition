import tensorflow as tf
from tensorflow.keras.layers import Layer, ZeroPadding2D, BatchNormalization, Conv2D
import numpy as np

from models.yolo.blocks import ConvBlock, ResBlock


class DarkNet53(Layer):
    def __init__(self):
        self._conv1 = ConvBlock(filter_shape=(3, 3,  3,  32))
        self._conv2 = ConvBlock(filter_shape=(3, 3,  32,  64), downsample=True)

        self._res_blocks1 = [
            ResBlock(input_channel=64, filter_conv1=32, filter_conv2=64)]

        self._conv3 = ConvBlock(filter_shape=(3, 3,  64, 128), downsample=True)

        self._res_blocks2 = []
        for i in range(2):
            self._res_blocks2.append(
                ResBlock(input_channel=128, filter_conv1=64, filter_conv2=128))

        self._conv4 = ConvBlock(filter_shape=(3, 3, 128, 256), downsample=True)

        self._res_blocks3 = []
        for i in range(8):
            self._res_blocks3.append(
                ResBlock(input_channel=356, filter_conv1=128, filter_conv2=256))

        self._conv5 = ConvBlock(filter_shape=(3, 3, 256, 512), downsample=True)

        self._res_blocks4 = []
        for i in range(8):
            self._res_blocks3.append(
                ResBlock(input_channel=512, filter_conv1=256, filter_conv2=512))

        self._conv6 = ConvBlock(filter_shape=(
            3, 3, 512, 1024), downsample=True)

        self._res_blocks5 = []
        for i in range(4):
            self._res_blocks3.append(
                ResBlock(input_channel=1024, filter_conv1=512, filter_conv2=1024))

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self._conv1(x)
        x = self._conv2(x)
        for b in self._res_blocks1:
            x = b(x)
        x = self._conv3(x)
        for b in self._res_blocks2:
            x = b(x)
        x = self._conv4(x)
        for b in self._res_blocks3:
            x = b(x)

        route_1 = x

        x = self._conv5(x)
        for b in self._res_blocks4:
            x = b(x)

        route_2 = x

        x = self._conv6(x)
        for b in self._res_blocks5:
            x = b(x)

        return route_1, route_2, x
