import tensorflow as tf
from tensorflow.keras.layers import ZeroPadding2D, BatchNormalization, Conv2D, Add


def conv_block(inputs, filters, kernel_size, down_sample=False, activate=True, batch_norm=True):
    if down_sample:
        inputs = ZeroPadding2D(((1, 0), (0, 1)))(inputs)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding=padding,
               use_bias=not batch_norm,
               kernel_regularizer=tf.keras.regularizers.l2(5e-4),
               kernel_initializer=tf.random_normal_initializer(stddev=0.01),
               bias_initializer=tf.constant_initializer(0.))(inputs)

    if batch_norm:
        x = BatchNormalization()(x)
    if activate:
        x = tf.nn.leaky_relu(x, alpha=0.1)

    return x


def res_block(inputs, filter_conv1, filter_conv2, flag=False):
    if flag:
        x = conv_block(inputs, filters=filter_conv1, kernel_size=1)
        x = conv_block(x, filters=filter_conv2, kernel_size=3)
    else:
        x = conv_block(inputs, filters=filter_conv1, kernel_size=1)
        x = conv_block(x, filters=filter_conv2, kernel_size=3)

    res_out = tf.keras.layers.add([x, inputs])
    return res_out
