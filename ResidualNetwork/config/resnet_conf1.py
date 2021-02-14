import tensorflow as tf
from models.resnet import ResnetEncoder
from models.bilstm import BiLSTMDecoder
from models.encoder_decoder import EncoderDecoder
from data.disfluency import DisfluencyDataLayer
from losses.cross_entropy_loss import CrossEntropyLoss
from optimizers.lr_policies import exp_decay
from optimizers.lr_policies import piecewise_constant

base_model = EncoderDecoder

base_params = {

    # "load_model": "egs/librispeech/ds2_log",

    "random_seed": 0,
    "num_epochs": 30,
    
    "num_gpus": 1,
    "batch_size_per_gpu": 32,
    
    "restore_best_ckpt": False,    
    "save_summaries_steps": 40,
    "print_loss_steps": 40,
    "print_samples_steps": 40,
    "eval_steps": 40,
    "save_checkpoint_steps": 40,
    "logdir": "logdir/tuning1",

    "optimizer": "Momentum",
    "optimizer_params": {
        "momentum": 0.90,
    },
    "lr_policy": piecewise_constant,
    "lr_policy_params": {
        "learning_rate": 0.0001,
        "boundaries":  [30, 60, 80, 90],
        "decay_rates": [0.1, 0.01, 0.001, 0.0001],
    },

    "dtype": tf.float32,
    "regularizer": tf.contrib.layers.l2_regularizer,
    "regularizer_params": {"scale": 0.0001},

    "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
                  'variable_norm', 'gradients_norm', 'gloabl_gradient_norm'],

    "initializer": tf.contrib.layers.xavier_initializer,
    
    "encoder": ResnetEncoder,
    "encoder_params":{

        "conv_layer0":{
            "kernel_size": 7,
            "filters": 64,
            "strides": 1
        },

        "block_size": 6,
        
        "res_block1":{
            "filters": [32, 64, 64],
            "strides": [1, 1, 2],
        },
        "res_block2":{
            "filters": [64, 128, 128],
            "strides": [1, 1, 2],
        },
        "res_block3":{
            "filters": [128, 128, 128],
            "strides": [1, 1, [2,1]],
        },
        "res_block4":{
            "filters": [128, 64, 64],
            "strides": [1, 1, 2],
        },
        "res_block5":{
            "filters": [64, 64, 32],
            "strides": [1, 1, 2],
        },
        "res_block6":{
            "filters": [32, 16, 16],
            "strides": [1, 1, 2],
        },
       
        "regularizer_bn": False,
        "data_format": "channels_last", # batch first channel last
        "dtype": tf.float32,
    },
        
    "decoder":BiLSTMDecoder,
    "decoder_params":{
        "n_hidden": 1024,
        "rnn_cell_dim": 512,
        "rnn_type": "lstm",
        "num_rnn_layers": 2,
        "rnn_unidirectional": False,
        "use_cudnn_rnn": False,
        "dropout": 0.5,
        "output_dim": 2
    },

    "loss": CrossEntropyLoss,
    "loss_params": {},
    "data_layer": DisfluencyDataLayer,
    "data_layer_params":{
        "dataset": "dev",
        "set": "A",
        "detection": "negative",
        "num_audio_features": 257,
        "shuffle": True,
    },
}
