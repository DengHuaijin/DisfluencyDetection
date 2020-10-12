import os
import sys
import h5py
import pickle
import argparse
import datetime
import numpy as np
import tensorflow as tf
import tensorflow.keras.utils as k_utils

from Config import Config
from data_process import PROCESS
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import backend as K
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras import Input, optimizers, Model, callbacks, regularizers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Lambda, Conv2D, MaxPooling2D, Reshape
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Bidirectional, PReLU
from tensorflow.keras.layers import Dropout, concatenate, Masking, LeakyReLU, Activation, dot
from tensorflow.keras.models import load_model as keras_load_model

parser = argparse.ArgumentParser()
parser.add_argument("--detection", default = None, required = True)
parser.add_argument("--gpu", default = "0")
parser.add_argument("--dataset", default = "dev")
args = parser.parse_args()

detection = args.detection
dataset = args.dataset
gpu = args.gpu

os.environ["CUDA_VISIBLE_DEVICES"] = gpu

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
K.set_session(sess)

HP_STRIDES = hp.HParam("strides", hp.Discrete([1,2,3]))
HP_DROPOUT = hp.HParam("cnn_dropout", hp.Discrete([0.3, 0.4, 0.5]))
# HP_FILTER = hp.HParam("filter_size", hp.Discrete([3,3],[2,2]))
METRIC_ACCURACY = "accuracy"

hp.hparams_config(
        hparams = [HP_STRIDES, HP_DROPOUT],
        metrics = [hp.Metric(METRIC_ACCURACY, display_name = "Accuarcy")]
)

def load_data(num_set):
    
    process = PROCESS(num_set = num_set)
    
    x_train_svm, x_train_lstm, y_train, x_test_svm, x_test_lstm, y_test = process.dataset_split(detection = detection, dataset = dataset)
    
    x_train_lstm = pad_sequences(x_train_lstm, padding = "post", dtype = np.float32, value = Config.MASKING_VALUE)
    x_test_lstm = pad_sequences(x_test_lstm, padding = "post", dtype = np.float32)
    
    class_weights = compute_class_weight(class_weight = "balanced", classes = np.unique(y_train), y = y_train)
    class_weight = {0: class_weights[0], 1: class_weights[1]}
    
    x_train_svm = x_train_svm[:, Config.SVM_FEATURE_SET]
    x_test_svm = x_test_svm[:, Config.SVM_FEATURE_SET]
    
    x_train_lstm = np.expand_dims(x_train_lstm, axis = -1)
    x_test_lstm = np.expand_dims(x_test_lstm, axis = -1)
    
    input_dims = x_train_lstm.shape[2]
    time_steps = x_train_lstm.shape[1]
    svm_dims = x_train_svm.shape[1]
    
    # one-hot
    y_train = k_utils.to_categorical(y_train)
    y_val = k_utils.to_categorical(y_test)
    
    x_train_dict = {'lstm_features': x_train_lstm, 'svm_features': x_train_svm}
    x_val_dict = {'lstm_features':x_test_lstm, 'svm_features': x_test_svm}
    
    train_set = tf.data.Dataset.from_tensor_slices((x_train_dict, y_train)).shuffle(buffer_size = Config.BUFFER_SIZE).batch(Config.BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    val_set = tf.data.Dataset.from_tensor_slices((x_val_dict, y_val)).shuffle(buffer_size = Config.BUFFER_SIZE).batch(Config.BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    return train_set, val_set, input_dims, time_steps, svm_dims, class_weight 

def train_test_model(hparams, num_set, session_num):

    train_set, val_set, input_dims, time_steps, svm_dims, class_weight = load_data(num_set = num_set)

    with K.name_scope("SVM_features"):

        svm_features = Input(shape = (svm_dims,), name = "svm_features")
        svm_input = Dense(128, activation = None, name = "svm_dense")(svm_features)
        svm_input = LeakyReLU()(svm_input)
    
    with K.name_scope("CNN"):
                
        lstm_features = Input(shape = (None, input_dims, 1), name = "lstm_features")
        # lstm_skip = Lambda(lambda t: t[:, 0:-1:2, :])(lstm_features)
        lstm_mask = Masking(mask_value = Config.MASKING_VALUE, input_shape = (time_steps, input_dims, 1))(lstm_features) # [None, T, F]
        lstm_mask = Lambda(lambda t: t)(lstm_mask)
        
        conv1 = Conv2D(16, hparams["HP_FILTER"], strides = 2, padding = "same", name = "conv1")(lstm_mask)
        conv1 = BatchNormalization()(conv1)
        if hparams["HP_FUNC"] == "tanh":
            conv1 = Activation("tanh")(conv1)
        if hparams["HP_FUNC"] == "relu":
            conv1 = Activation("relu")(conv1)
        if hparams["HP_FUNC"] == "leakyrelu":
            conv1 = LeakyReLU()(conv1)
        conv1_drop = Dropout(rate = hparams["HP_DROP"])(conv1)
        # conv1_pool = MaxPooling2D(pool_size = (1,2), stride = 1, name = "maxpooling1")(conv1)

        conv2 = Conv2D(32, hparams["HP_FILTER"], strides = 2, padding = "same", name = "conv2")(conv1_drop)
        conv2 = BatchNormalization()(conv2)
        if hparams["HP_FUNC"] == "tanh":
            conv2 = Activation("tanh")(conv2)
        if hparams["HP_FUNC"] == "relu":
            conv2 = Activation("relu")(conv2)
        if hparams["HP_FUNC"] == "leakyrelu":
            conv2 = LeakyReLU()(conv2)
        conv2_drop = Dropout(rate = hparams["HP_DROP"])(conv2)
        # conv2_pool = MaxPooling2D(pool_size = (1,2), stride = 1, name = "maxpooling1")(conv2)
        
        conv_reshape = Lambda(lambda t: tf.concat(tf.unstack(t, axis = -1), axis = -1))(conv2_drop)
        # conv_reshape = Reshape(target_shape = (-1, 16))(conv1_pool)

    with K.name_scope("LSTM"):
        
        # conv_dense = Dense(16, activation = None, name = "conv_dense")(conv_reshape)
        # conv_dense = LeakyReLU()(conv_reshape)
        lstm_output = LSTM(Config.LSTM_UNITS, return_sequences = False, name = "lstm_sequence")(conv_reshape)
        # lstm_output_pool = Lambda(lambda t: tf.reduce_mean(t, 1))(lstm_output)
    
    with K.name_scope("Concatenate"):
        x = concatenate([lstm_output, svm_input])
        x_dense = Dense(128, activation = None)(x)
        x_dense = LeakyReLU()(x_dense)
        # batchnorm1 = BatchNormalization()(x)
        # dropout1 = Dropout(rate = 0.3)(batchnorm1)
        dense_2 = Dense(128, activation = None)(x_dense)
        dense_2 = LeakyReLU()(dense_2)
        batchnorm2 = BatchNormalization()(dense_2)
        dropout = Dropout(rate = 0.3)(batchnorm2) 
        
    pred = Dense(2, activation = "softmax", name = "output")(dropout)
    model = Model(inputs = [svm_features, lstm_features], outputs = [pred])
   
    logdir = os.path.join("tensorboard_log", detection, "tuning_5", str(session_num), num_set)

    if os.path.exists(logdir):
        raise ValueError("The tuning set existed")

    else:
        cmd = "mkdir -p " + logdir
        os.system(cmd)
    
    tensorboard_callback = callbacks.TensorBoard(log_dir = logdir, write_graph = False)

    model.compile(loss = "binary_crossentropy",
                  optimizer = optimizers.Adam(lr = 1e-5),
                  metrics = ["accuracy"])
    
    history = model.fit(x = train_set,
                        validation_data = val_set,
                        epochs = 50,
                        verbose = 0,
                        class_weight = class_weight,
                        callbacks = [tensorboard_callback])
    
    loss = history.history["val_loss"][-1]

    return loss

def run(run_dir, hparams, session_num):

    # hp.hparams(hparams)
    tmp = 0
    for num_set in ["A", "B", "C", "D", "E"]:
        print("-------------------set {}------------------".format(num_set))
        loss = train_test_model(hparams, num_set, session_num)
        tmp = loss + tmp
    
    loss = tmp / 5

    print(loss)
    # tf.summary.scalar(METRIC_ACCURACY, accuracy, step = 1)

session_num = -1
for filter_size in [[3,3], [4,4], [5,5], [30,2], [40,2]]:
    for drop in [0.3, 0.4, 0.5]:
        for func in ["tanh", "relu", "leakyrelu"]:
                
            session_num += 1
            if session_num < 26:
                continue
                
            hparams = {
                    "HP_DROP": drop,
                    "HP_FUNC": func,
                    "HP_FILTER": filter_size
                    }
            
            run_name = "run-{}".format(session_num)
            print("----starting trial: {}".format(run_name))
            print({h: hparams[h] for h in hparams})
            run("log/hparam_tuning/" + run_name, hparams, session_num)

