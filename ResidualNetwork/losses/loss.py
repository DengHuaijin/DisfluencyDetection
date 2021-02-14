from __future__ import absolute_import, print_function, division
from __future__ import unicode_literals

import abc
import copy

import six
import tensorflow as tf

from utils.utils import check_params, cast_types

@six.add_metaclass(abc.ABCMeta)
class Loss:
    
    @staticmethod
    def get_required_params():
        return {}

    @staticmethod
    def get_optional_params():
        return {
                "dtype": [tf.float16, tf.float32]
               }

    def __init__(self, params, model, name = "loss"):
        
        check_params(params, self.get_required_params(), self.get_optional_params())
        self._params = copy.deepcopy(params)
        self._model = model

        if "dtype" not in self._params:
            if self._model:
                self._params["dtype"] = self._model.get_tf_dtype()
            else:
                self._params["dtype"] = tf.float32

        self._name = name


    def compute_loss(self, input_dict):

        with tf.variable_scope(self._name, dtype = self.params["dtype"]):
            return self._compute_loss(self._cast_types(input_dict))

    def _cast_types(self, input_dict):

        return cast_types(input_dict, self.params["dtype"])

    @abc.abstractmethod
    def _compute_loss(self, input_dict):
        pass

    @property
    def params(self):
        return self._params

    @property
    def name(self):
        return self._name

