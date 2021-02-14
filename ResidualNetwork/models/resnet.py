from __future__ import absolute_import, print_function
from __future__ import division
from __future__ import unicode_literals

import os
import tensorflow as tf

from models.model import Model
from models.resnet_block import conv2d_fixed_padding, batch_norm, building_block_v1
from utils.utils import deco_print

class ResnetEncoder:
    
    @staticmethod
    def get_required_params():
        return dict(Model.get_required_params(), **{})

    @staticmethod
    def get_optional_params():
        return dict(Model.get_optional_params(), **{})

    def __init__(self, params, mode = "train"):
        self.mode = params["mode"]
        self.params = params
        # super(ResnetEncoder, self).__init__(params, mode)

    def _encode(self, input_dict):
        
        inputs = input_dict["source_tensors"][0]

        print("\nOriginal batch input: {}".format(inputs))

        inputs = tf.expand_dims(inputs, axis = -1)
        inputs = tf.cast(inputs, self.params["dtype"])
        batch_size = inputs.get_shape().as_list()[0]

        data_format = self.params.get("data_format", "channels_first")
        bn_momentum = self.params.get("bn_momentum", 0.997)
        bn_epsilon = self.params.get("bn_epsilon", 1e-5)

        training = self.mode == "train"
        regularizer = self.params.get("regularizer", None)
        regularizer_bn = self.params.get("regularizer_bn", True)
        bn_regularizer = regularizer if regularizer_bn else None
        
        if data_format == "channels_first":
            inputs = tf.transpose(inputs, [0,3,1,2])
        
        print("\nInput data: {}".format(inputs))
        ##### initial conv layer #####
        # inputs: [batch_size, 200, 257, 1]
        with tf.variable_scope("initConvLayer"):
            conv0_filters = self.params["conv_layer0"]["filters"]
            conv0_kernel_size = self.params["conv_layer0"]["kernel_size"]
            conv0_strides = self.params["conv_layer0"]["strides"]
            inputs = conv2d_fixed_padding(inputs = inputs, filters = conv0_filters,
                                          kernel_size = conv0_kernel_size, 
                                          strides = conv0_strides, data_format = data_format,
                                          regularizer = regularizer)
            inputs = tf.identity(inputs, "initial_conv")
            inputs = batch_norm(inputs, training, data_format, 
                                regularizer = bn_regularizer, 
                                momentum = bn_momentum, epsilon = bn_epsilon)
            inputs = tf.nn.relu(inputs)
        print("\nInit conv layer output: {}\n".format(inputs))
        #####
       
        with tf.variable_scope("residualBlock"):
            block_size = self.params["block_size"]
            # block_strides = self.params["block_strides"]
            for i in range(block_size):
                cur_block = "res_block" + str(i+1)
                cur_filters = self.params[cur_block]["filters"]
                cur_strides = self.params[cur_block]["strides"]
                cur_name = "block_layer" + str(i+1)
                inputs = building_block_v1(inputs = inputs, filters = cur_filters, training = training,
                                           projection_shortcut = True, strides = cur_strides, 
                                           data_format = data_format, regularizer = regularizer,
                                           bn_regularizer = bn_regularizer, bn_momentum = bn_momentum,
                                           bn_epsilon = bn_epsilon, name = cur_name)
                print("block {} output: {}".format(i+1, inputs))
        
        f = inputs.get_shape().as_list()[2]
        c = inputs.get_shape().as_list()[3]
        # axes = [2, 3] if data_format == "channels_first" else [1, 2]
        # inputs = rf.reduce_mean(inputs, axes, keepdims = True)
        outputs = tf.reshape(inputs, [batch_size, -1, f*c])
        print("CNN fully connected output: {} \n".format(outputs))
        # outputs = tf.reshape(inputs, [-1, self.params["final_size"]])

        return {"outputs": outputs}

