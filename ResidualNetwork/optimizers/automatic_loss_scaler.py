from __future__ import absolute_import, print_function, division
from __future__ import unicode_literals

import tensorflow as tf

from asr_e2e.utils.utils import check_params

class AutomaticLossScaler(object):

    SUPPORT_ALGOS = ["backoff", "logmax"]

    def __init__(self, algorithm = "Backoff", params = None):
        algorithm = algorithm.lower().strip()
        if algorithm == "backoff":
            self.scaler = Backoff(params)
        elif algorithm == "logmax":
            self.scaler = LogMaxScaler(params)
        else:
            raise ValueError("Unknown scaling algorithm: {}".format(algorithm))

    def update_op(self, has_nan, amax):
        return self.scaler.update_op(has_nan, amax)

    @property
    def loss_scale(self):
        return self.scaler.loss_scale

    @staticmethod
    def check_grads(grads_and_vars):
        has_nan_ops = []
        amax_ops = []

        for grad, _ in grads_and_vars:
            if grad is not None:
                if isinstance(grad, tf.IndexSlices):
                    x = grad.values
                else:
                    x = grad

                has_nan_ops.append(tf.reduce_any(tf.is_nan(x)))
                amax_ops.append(tf.reduce_max(tf.abs(x)))

        has_nan = tf.reduce_any(has_nan_ops)
        amax = tf.reduce_max(amax_ops)

        return has_nan, amax
