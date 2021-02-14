from __future__ import absolute_import, print_function, division
from __future__ import unicode_literals

import os
import sys
import numpy as np
import tensorflow as tf
import six
import math
from six import string_types
from six.moves import range

from data_process import PROCESS
from data.data_layer import DataLayer

if hasattr(np.fft, "restore_all"):
    np.fft.restore_all()


class DisfluencyDataLayer(DataLayer):

    @staticmethod
    def get_required_params():
        return dict(DataLayer.get_required_params(), **{})
    
    @staticmethod
    def get_optional_params():
        return dict(DataLayer.get_optional_params(), **{
            "num_audio_features": int,
        }) 

    def __init__(self, params, model, num_workers, worker_id):
        
        super(DisfluencyDataLayer, self).__init__(params, model, num_workers, worker_id)

        self.target_pad_value = 0

        self._dataset = None
        self._iterator = None
        self._input_tensors = None

        data_process = PROCESS(num_set = self.params["set"], scaling = 0)
        _, x_train_spec, y_train, _, x_test_spec, y_test = data_process.dataset_split(detection = self.params["detection"],
                                                                                      dataset = self.params["dataset"])
        if self.params["mode"] == "train":
            self.x_spec = x_train_spec
            self.y = y_train
        elif self.params["mode"] == "eval":
            self.x_spec = x_test_spec
            self.y = y_test
        elif self.params["mode"] == "infer":
            self.x_spec = x_test_spec
            self.y = y_test

        self._size = len(self.x_spec)
         
        self.x_spec = tf.keras.preprocessing.sequence.pad_sequences(self.x_spec, 
                                                                    maxlen = 500,
                                                                    dtype = "float",
                                                                    padding = "post",
                                                                    value = 0.0)
        def one_hot(label):
            tmp = [0, 0]
            if int(label) == 0:
                return [1, 0]
            else:
                return [0, 1]

        self.y = list(map(one_hot, self.y))
        # self.tfrecord_file = os.path.join("data", "tfrecord", self.params["set"], "tmp.tfrecord")
        # self.write_tfrecord(specs = list(self.x_spec), labels = self.y, filename = self.tfrecord_file)

    def split_data(self, data):

        if self.params["mode"] != "train" and self._num_workers is not None:
            size = len(data)
            """
            多GPU计算时，以3GPU size=9为例：
            每个GPU等分数据量
            GPU1: [0:3]
            GPU2: [3:6]
            GPU3: [6:9]
            """
            start = size // self._num_workers * self._worker_id
            if self._worker_id == self._num_workers - 1:
                end = size
            else:
                end = size // self._num_workers * (self._worker_id + 1)

            return data[start:end]
        else:
            return data

    @property
    def iterator(self):
        return self._iterator

    def build_graph(self):
        
        with tf.device("/cpu:0"):
            """
            Builds data processing graph using tf.data API
            """
            if self.params["mode"] != "infer":
                self._dataset = tf.data.Dataset.from_tensor_slices((self.x_spec, self.y))
                if self.params["shuffle"]:
                    self._dataset = self._dataset.shuffle(self._size)

                self._dataset = self._dataset.repeat()
                self._dataset = self._dataset.prefetch(tf.contrib.data.AUTOTUNE)
                self._dataset = self._dataset.batch(self.params["batch_size"])
                
            else:
                #TODO
                self._dataset = tf.data.Dataset.from_tensor_slices()
                self._dataset = self._dataset.repeat()
                self._dataset = self._dataset.prefetch(tf.contrib.data.AUTOTUNE)

                self._dataset = self._dataset.padded_batch(
                        self.params["batch_size"],
                        padded_shapes = ([None, self.params["num_audio_features"]]))
            
            self._iterator = self._dataset.prefetch(tf.contrib.data.AUTOTUNE).make_initializable_iterator()
            
            if self.params["mode"] != "infer":
                x_spec, y = self._iterator.get_next()
                y = tf.reshape(y, [self.params["batch_size"], 2])
            else:
                x_train_spec = self._iterator.get_next()
            
            # [B,T,F]
            x_spec.set_shape([self.params["batch_size"], None, self.params["num_audio_features"]])

            self._input_tensors = {}
            self._input_tensors["source_tensors"] = [x_spec]
            
            if self.params["mode"] != "infer":
                self._input_tensors["target_tensors"] = [y]
            else:
                self._input_tensors["source_target"] = []


    def get_size_in_samples(self):
        return len(self.x_spec)
    

    def create_feed_dict(self, model_input):
        """
        model_input: a str that contains the path of the wav file or a 1-d array containing 1-d wav file
        """
        audio_array = []
        audio_length_array = []
        x_id_array = []

        for line in model_input:
            if isinstance(line, string_types):
                audio, audio_length, x_id, _ = self._parse_audio_element([0, line])
            elif isinstance(line, np.ndarray):
                audio, audio_length, x_id, _ = self._get_audio(line)
            else:
                raise ValueError("Interactive mode only supports string or numpy array. Got {}".format(type(line)))

            audio_array.append(audio)
            audio_length_array.append(audio_length)
            x_id_array.append(x_id)

        max_len = np.max(audio_length_array)
        pad_to = self.params.get("pad_to", 8)
        if pad_to > 0 and self.params.get("backend") == "librosa":
            max_len += (pad_to - max_len % pad_to) % pad_to
        for i, audio in enumerate(audio_array):
            audio = np.pad(audio, ((0, max_len - len(audio)), (0, 0)), "constant", constant_values = 0)
            audio_array[i] = audio
        
        audio = np.reshape(audio_array,
                        [self.params["batch_size"],
                        -1,
                        self.params["num_audio_features"]])

        audio_length = np.reshape(audio_length_array, [self.params["batch_size"]])
        x_id = np.reshape(x_id_array, [self.params["batch_size"]])

        feed_dict = {
                self._x: audio,
                self._x_length: audio_length,
                self._x_id: x_id}

        return feed_dict

    def create_interactive_placeholders(self):
        self._x = tf.placeholder(
                dtype = self.params["dtype"],
                shape = [
                    self.params["batch_size"],
                    None,
                    self.params["num_audio_features"]
                    ]
                )
        
        self._x_length = tf.placeholder(
                dtype = tf.int32,
                shape = [self.params["batch_size"]])

        self._x_id = tf.placeholder(
                dtype = tf.int32,
                shape = [self.params["batch_size"]])

        self._input_tensors = {}
        self._input_tensors["source_tensors"] = [self._x, self._x_length]
        self._input_tensors["source_ids"] = [self._x_id]

    @property
    def input_tensors(self):
        return self._input_tensors

    def write_tfrecord(self, specs, labels, filename):
        writer = tf.python_io.TFRecordWriter(filename)
        for feature, label in zip(specs, labels):
            # print(len(feature.ravel().tolist()))
            # sys.exit(0)
            example = tf.train.Example(features = tf.train.Features(feature = {
                "spectrogram": tf.train.Feature(float_list = tf.train.FloatList(value = [feature.ravel().tolist()])),
                "label": tf.train.Feature(int64_list = tf.train.Int64List(value = [int(label)]))
                }))
            writer.write(example.SerializeToString())
        writer.close()
