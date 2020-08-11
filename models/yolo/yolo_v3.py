import tensorflow as tf
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.models import Model

from models.yolo.blocks import ConvBlock
from models.yolo.darknet53 import DarkNet53


class YoloV3(Model):
    def __init__(self):
        self._backbone = DarkNet53()

        self._conv1 = ConvBlock(filter_shape=(1, 1, 1024, 512))
        self._conv2 = ConvBlock(filter_shape=(3, 3, 512, 1024))
        self._conv3 = ConvBlock(filter_shape=(1, 1, 1024, 512))
        self._conv4 = ConvBlock(filter_shape=(3, 3, 512, 1024))
        self._conv5 = ConvBlock(filter_shape=(1, 1, 1024, 512))

        self._conv_lobj = ConvBlock(filter_shape=(3, 3, 512, 1024))
        # self._conv_lbbox = ConvBlock(filter_shape=(
        #     1, 1, 1024, 3*(NUM_CLASS + 5)), activate=False, batch_norm=False)
        self._conv_lbbox = ConvBlock(filter_shape=(
            1, 1, 1024, 5), activate=False, batch_norm=False)

        self._conv6 = ConvBlock(filter_shape=(1, 1, 512, 256))
        self._upsample1 = UpSampling2D()

        self._conv7 = ConvBlock(filter_shape=(1, 1, 768, 256))
        self._conv8 = ConvBlock(filter_shape=(3, 3, 256, 512))
        self._conv9 = ConvBlock(filter_shape=(1, 1, 512, 256))
        self._conv10 = ConvBlock(filter_shape=(3, 3, 256, 512))
        self._conv11 = ConvBlock(filter_shape=(1, 1, 512, 256))

        self._conv_mobj = ConvBlock(filter_shape=(3, 3, 256, 512))
        # self._conv_mbbox = ConvBlock(filter_shape=(
        #     1, 1, 512, 3*(NUM_CLASS + 5)), activate=False, batch_norm=False)
        self._conv_mbbox = ConvBlock(filter_shape=(
            1, 1, 512, 5), activate=False, batch_norm=False)

        self._conv12 = ConvBlock(filter_shape=(conv, (1, 1, 256, 128)))
        self._upsample2 = UpSampling2D()

        self._conv13 = ConvBlock(filter_shape=(1, 1, 384, 128))
        self._conv14 = ConvBlock(filter_shape=(3, 3, 128, 256))
        self._conv15 = ConvBlock(filter_shape=(1, 1, 256, 128))
        self._conv16 = ConvBlock(filter_shape=(3, 3, 128, 256))
        self._conv17 = ConvBlock(filter_shape=(1, 1, 256, 128))

        self._conv_sobj = ConvBlock(filter_shape=(3, 3, 128, 256))
        # self._sbbox = ConvBlock(filter_shape=(
        #     1, 1, 256, 3*(NUM_CLASS + 5)), activate=False, bn=False)
        self._conv_sbbox = ConvBlock(filter_shape=(1, 1, 256, 5), activate=False, batch_norm=False)

    def call(self, x, training=None, mask=None):
        route_1, route_2, x = self._backbone(x)

        x = self._conv1(x)
        x = self._conv2(x)
        x = self._conv3(x)
        x = self._conv4(x)
        x = self._conv5(x)

        conv_lobj = self._conv_lobj(x)
        conv_lbbox = self._conv_lbbox(conv_lobj)

        x = self._conv6(x)
        x = self._upsample1(x)

        x = tf.concat([x, route_2], axis=-1)

        x = self._conv7(x)
        x = self._conv8(x)
        x = self._conv9(x)
        x = self._conv10(x)
        x = self._conv11(x)

        conv_mobj = self._conv_mobj(x)
        conv_mbbox = self._conv_mbbox(conv_mobj)

        x = self._conv12(x)
        x = self._upsample2(x)

        x = tf.concat([x, route_1], axis=-1)

        x = self._conv13(x)
        x = self._conv14(x)
        x = self._conv15(x)
        x = self._conv16(x)
        x = self._conv17(x)

        conv_sobj = self._conv_sobj(x)
        conv_sbbox = self._conv_sbbox(conv_sobj)

        return [conv_sbbox, conv_mbbox, conv_lbbox]
