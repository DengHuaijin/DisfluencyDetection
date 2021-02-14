from __future__ import absolute_import, print_function, division
from __future__ import unicode_literals

import os
import sys
import tensorflow as tf

from models.model import Model
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops

class BiLSTMDecoder:
    
    @staticmethod
    def get_required_params():
        return dict(Model.get_required_params(), **{
            "output_dim": int,
            })
    
    def __init__(self, params, mode = "train"):
        self.mode = params["mode"]
        self.params = params
        # super(BiLSTMDecoder, self).__init__(params, model, name, mode)
    
    def rnn_cell(self, rnn_cell_dim, layer_type, dropout_keep_prob = 1.0):

        if layer_type == "lstm":
            cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_cell_dim)
        else:
            raise ValueError("Error: not suppported rnn type: {}".format(layer_type))

        cell = tf.nn.rnn_cell.DropoutWrapper(
                cell, output_keep_prob = dropout_keep_prob)
        
        return cell

    def _decode(self, input_dict):
        inputs = input_dict["encoder_output"]["outputs"]
        regularizer = self.params.get("regularizer", None)
        dropout = self.params["dropout"]
        """
        logits = tf.layers.dense(
                inputs = inputs,
                units = self.params["n_hidden"],
                kernel_regularizer = regularizer,
                name = "fully_connected1")
        """
        # [B, T, FC]
        #TODO

        num_rnn_layers = self.params["num_rnn_layers"]
        if num_rnn_layers > 0:
            rnn_cell_dim = self.params["rnn_cell_dim"]
            rnn_type = self.params["rnn_type"]
            if self.params["use_cudnn_rnn"]:
                rnn_input = tf.transpose(inputs, [1, 0, 2])
                if self.params["rnn_unidirectional"]:
                    direction = cudnn_rnn_ops.CUDNN_RNN_UNIDIRECTION
                else:
                    direction = cudnn_rnn_ops.CUDNN_RNN_BIDIRECTION

                rnn_block = tf.contrib.cudnn_rnn.CudnnLSTM(
                        num_layers = num_rnn_layers,
                        num_units = rnn_cell_dim,
                        direction = direction,
                        dropout = dropout,
                        dtype = rnn_input.dtype,
                        name = "cudnn_lstm")
            
                sequence, final_state = rnn_block(rnn_input, sequence_lengths = None, time_major = True)
                final_state = tf.concat(tf.slice(final_state[0], [2,0,0], [2,32,512]), 2)[0]
                print(sequence)
            else:
                rnn_input = inputs
                multirnn_cell_fw = tf.nn.rnn_cell.MultiRNNCell(
                        [self.rnn_cell(rnn_cell_dim = rnn_cell_dim, layer_type = rnn_type, dropout_keep_prob = 1 - dropout)
                        for _ in range(num_rnn_layers)]
                        )
                
                if self.params["rnn_unidirectional"]:
                    sequence, final_state = tf.nn.dynamic_rnn(
                            cell = multirnn_cell_fw,
                            inputs = rnn_input,
                            sequence_length = 4,
                            dtype = rnn_input.dtype,
                            time_major = False)
                else:
                    multirnn_cell_bw = tf.nn.rnn_cell.MultiRNNCell(
                            [self.rnn_cell(rnn_cell_dim = rnn_cell_dim, layer_type = rnn_type,
                                     dropout_keep_prob = 1 - dropout)
                            for _ in range(num_rnn_layers)]
                            )
                    
                    sequence, final_state = tf.nn.bidirectional_dynamic_rnn(
                            cell_fw = multirnn_cell_fw, cell_bw = multirnn_cell_bw,
                            inputs = rnn_input, sequence_length = None,
                            dtype = rnn_input.dtype,
                            time_major = False
                            )
                    final_state = tf.concat([final_state[0][-1], final_state[1][-1]], 2)[0]
                    print("BiLSTM last hidden state output: {}\n".format(final_state))
                
        outputs = tf.layers.dense(
                inputs = final_state,
                units = self.params["output_dim"],
                kernel_regularizer = regularizer,
                # activation = self.params["actiavation_fn"],
                name = "fully_connected")
        print("Final fully connected layer output: {}\n".format(outputs))
        sys.exit(0)     
        return {"logits": outputs, "outputs": [outputs]}
        
