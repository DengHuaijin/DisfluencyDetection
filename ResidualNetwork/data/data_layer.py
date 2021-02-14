from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import abc
import copy

import six
import tensorflow as tf

from utils.utils import check_params

@six.add_metaclass(abc.ABCMeta)
class DataLayer:

    @staticmethod
    def get_required_params():
        return {"mode": ["train", "eval", "infer"],
                "detection": str,
                "dataset": str,
                "set": str,
                }

    @staticmethod
    def get_optional_params():
        return {
                "batch_size": int,
                "shuffle": bool,
                "repeat": bool,
                "dtype": [tf.float32, tf.float16],
                "interactive": bool,
                "cache_format": str,
                "cache_regenerate": bool
                }

    @abc.abstractmethod
    def __init__(self, params, model, num_workers, worker_id):
        
        check_params(params, self.get_required_params(), self.get_optional_params())
        self._params = copy.deepcopy(params)
        self._model = model

        if "dtype" not in self._params:
            if self._model:
                self._params["dtype"] = self._model.get_tf_dtype()
            else:
                self._params["dtype"] = tf.float32

        if "shuffle" not in self._params:
            self._params["dtype"] = (self._params["mode"] == "train")

        if self._params["mode"] != "train" and self._params["shuffle"]:
            raise ValueError("Shuffle should not be performed in {} mode".format(self._params["mode"]))
        
        self._num_workers = num_workers
        self._worker_id = worker_id

    @property
    def params(self):
        return self._params

    @abc.abstractmethod
    def build_graph(self):
        pass

    @property
    @abc.abstractmethod
    def iterator(self):
        """
        tf.data.Dataset iterator
        should be created by self.build_graph()
        """
        pass

    @property
    @abc.abstractmethod
    def input_tensors(self):
        """
        keys: 'source_tensors' input sequence and input length
        'target_tensors' target sequence and target length
        all tensors have to be created inside build_graph()
        """
        pass

    def get_size_in_samples(self):
        """
        Should return the dataset size in samples.
        """
        return None

    def create_feed_dict(self, model_input):
        pass

    def create_interactive_placeholders(self):
        pass
