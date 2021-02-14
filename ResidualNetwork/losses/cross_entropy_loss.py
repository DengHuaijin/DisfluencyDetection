from __future__ import absolute_import, print_function, division
from __future__ import unicode_literals

import tensorflow as tf

from losses.loss import Loss

class CrossEntropyLoss(Loss):

    def __init__(self, params, model, name = "cross_entropy_loss"):
        super(CrossEntropyLoss, self).__init__(params, model, name)

    def _compute_loss(self, input_dict):
        
        logits = input_dict["decoder_output"]["logits"]
        labels = input_dict["target_tensors"][0]

        return tf.losses.softmax_cross_entropy(logits = logits, onehot_labels = labels)
