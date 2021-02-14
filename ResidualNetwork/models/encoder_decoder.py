from __future__ import absolute_import, print_function, division
from __future__ import unicode_literals

import tensorflow as tf
import numpy as np

from models.model import Model
from models.resnet import ResnetEncoder
from models.bilstm import BiLSTMDecoder
from utils.utils import deco_print

class EncoderDecoder(Model):

    @staticmethod
    def get_required_params():
        return dict(Model.get_required_params(), **{
            "encoder": None,
            "decoder": None,
        })

    @staticmethod
    def get_optional_params():
        return dict(Model.get_optional_params(), **{
            "encoder_params": dict,
            "decoder_params": dict,
            "loss": None,
            "loss_params": dict
        })

    def __init__(self, params, mode = "train", detection = "", dataset = "", num_set = ""):

        super(EncoderDecoder, self).__init__(params = params, mode = mode, detection = detection, dataset = dataset, num_set = num_set)
        
        self.params["encoder_params"]["mode"] = mode
        self.params["decoder_params"]["mode"] = mode
        self.encoder = ResnetEncoder(self.params["encoder_params"])
        self.decoder = BiLSTMDecoder(self.params["decoder_params"])
        
    def _build_forward_pass_graph(self, input_tensors, gpu_id = 0):

        if not isinstance(input_tensors, dict) or "source_tensors" not in input_tensors:
            raise ValueError("Input tensors should be a dict containing 'source_tensors' key")

        source_tensors = input_tensors["source_tensors"]
        if self.mode == "train" or self.mode == "eval":
            if "target_tensors" not in input_tensors:
                raise ValueError("Input tensors should contain 'target_tensors' key when mode == train or eval")
            target_tensors = input_tensors["target_tensors"]

        with tf.variable_scope("ForwardPass"):
            encoder_input = {"source_tensors": source_tensors}
            encoder_output = self.encoder._encode(input_dict = input_tensors)

            decoder_input = {"encoder_output": encoder_output}
            if self.mode == "train" or self.mode == "eval":
                decoder_input["target_tensors"] = target_tensors

            decoder_output = self.decoder._decode(input_dict = decoder_input)
            model_outputs = decoder_output["outputs"]
            
            if self.mode == "train" or self.mode == "eval":
                with tf.variable_scope("Loss"):
                    loss_input_dict = {
                            "decoder_output": decoder_output,
                            "target_tensors": target_tensors
                            }
                    loss_computor = self.params["loss"](params = self.params["loss_params"], model = self)
                    loss = loss_computor.compute_loss(loss_input_dict)
            
        return loss, model_outputs

    def maybe_print_logs(self, input_values, output_values, training_step):
        labels = input_values["target_tensors"][0]
        # one-hot labels
        labels = np.where(labels == 1)[1]
        logits = output_values[0]

        total = logits.shape[0]
        accuracy = 1.0 * np.sum(np.argmax(logits, axis = 1) == labels) / total

        deco_print("Train batch accuracy: {:.4f}".format(accuracy), offset = 4)

        return {"Train batch accuracy": accuracy}

    def _get_num_objects_per_step(self, work_id = 0):
        data_layer = self.get_data_layer(work_id)
        num_samples = tf.shape(data_layer.input_tensors["source_tensors"][0])[0]
        return num_samples



