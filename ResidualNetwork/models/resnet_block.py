from __future__ import absolute_import, print_function, division
from __future__ import unicode_literals

from six.moves import range

import tensorflow as tf

def batch_norm(inputs, training, data_format, regularizer, momentum, epsilon):

    return tf.layers.batch_normalization(
            inputs = inputs, axis = 1 if data_format == "channels_first" else 3,
            momentum = momentum, epsilon = epsilon, center = True,
            scale = True, training = training, fused = True, gamma_regularizer = regularizer)

def fixed_padding(inputs, kernel_size, data_format):

    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == "channels_first":
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])

    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])

    return padded_inputs

def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format, regularizer):
    """
    if strides is int and strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)
    """
    return tf.layers.conv2d(inputs = inputs, filters = filters, kernel_size = kernel_size,
                            strides = strides, padding = "SAME",
                            use_bias = False, data_format = data_format, kernel_regularizer = regularizer)

def building_block_v1(inputs, filters, training, projection_shortcut, strides, 
                      data_format, regularizer, bn_regularizer, bn_momentum,
                      bn_epsilon, name):
    
    shortcut = inputs
    if projection_shortcut:
        shortcut = conv2d_fixed_padding(inputs = shortcut, filters = filters[2], kernel_size = 3,
                                        strides = strides[2], data_format = data_format,
                                        regularizer = regularizer)
        shortcut = batch_norm(inputs = shortcut, training = training, 
                              data_format = data_format, regularizer = regularizer,
                              momentum = bn_momentum, epsilon = bn_epsilon)
    # conv1
    inputs = conv2d_fixed_padding(
            inputs = inputs, filters = filters[0], kernel_size = 3, strides = strides[0],
            data_format = data_format, regularizer = regularizer)
    inputs = batch_norm(inputs, training, data_format, regularizer = bn_regularizer,
                        momentum = bn_momentum, epsilon = bn_epsilon)
    inputs = tf.nn.relu(inputs)

    # conv2
    inputs = conv2d_fixed_padding(
            inputs = inputs, filters = filters[1], kernel_size = 3, strides = strides[0],
            data_format = data_format, regularizer = regularizer)
    inputs = batch_norm(inputs, training, data_format, regularizer = bn_regularizer,
                        momentum = bn_momentum, epsilon = bn_epsilon)
    inputs = tf.nn.relu(inputs)

    # conv3
    inputs = conv2d_fixed_padding(
            inputs = inputs, filters = filters[2], kernel_size = 3, strides = strides[2],
            data_format = data_format, regularizer = regularizer)
    inputs = batch_norm(inputs, training, data_format, regularizer = bn_regularizer,
                        momentum = bn_momentum, epsilon = bn_epsilon)

    inputs += shortcut
    inputs = tf.nn.relu(inputs)

    inputs = tf.identity(inputs, name)

    return inputs

def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides,
                training, name, data_format, regularizer, bn_regularizer,
                bn_momentum, bn_epsilon):
    
    filters_out = filters * 4 if bottleneck else filters

    def projection_shortcut(inputs):
        return conv2d_fixed_padding(inputs = inputs, filters = filters, kernel_size = 1,
                                    strides = strides, data_format = data_format,
                                    regularizer = regularizer)
