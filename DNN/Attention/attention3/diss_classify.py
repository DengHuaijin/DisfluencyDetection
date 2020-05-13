import os
import sys
import pydot_ng
import numpy as np
import tensorflow.keras.utils as k_utils

from Config import Config
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.sequence import pad_sequences
from dnn_model import LSTM_Model, DNN_Model
from data_process import PROCESS

def Train(save_model_name: str, num_set: str, detection: str, dataset: str, load = False):
    
    process = PROCESS(num_set = num_set)
    x_train_svm, x_train_lstm, y_train, x_test_svm, x_test_lstm, y_test = process.dataset_split(detection = detection, dataset = dataset)
    
    x_train_lstm = pad_sequences(x_train_lstm, padding = "post", dtype = np.float32, value = Config.MASKING_VALUE)
    x_test_lstm = pad_sequences(x_test_lstm, padding = "post", dtype = np.float32)
    
    # x_train_u = np.full((x_train_lstm.shape[0], Config.ATTENTION_UNITS), Config.ATTENTION_INIT, dtype = np.float32)
    # x_test_u = np.full((x_test_lstm.shape[0], Config.ATTENTION_UNITS), Config.ATTENTION_INIT, dtype = np.float32)

    class_weights = compute_class_weight(class_weight = "balanced", classes = np.unique(y_train), y = y_train)
    class_weight = {0: class_weights[0], 1: class_weights[1]}
    
    x_train_svm = x_train_svm[:, Config.SVM_FEATURE_SET]
    x_test_svm = x_test_svm[:, Config.SVM_FEATURE_SET]
    
    # one-hot
    y_train = k_utils.to_categorical(y_train)
    y_val = k_utils.to_categorical(y_test)

    print('------------------------------- Data Check --------------------------')
    print("SVM features: ", x_train_svm.shape, x_test_svm.shape)
    print("LSTM featrues: ", x_train_lstm.shape, x_test_lstm.shape)
    print("Sequential mean: ", np.mean(x_train_lstm[0, :, 1]), "Sequential std: ", np.std(x_train_lstm[0, :, 1]))
    print("SVM Train max: {} min: {}  Test max: {} min: {}".format(max(x_train_svm[:, 1]), min(x_train_svm[:, 1]), 
                                                                   max(x_test_svm[:, 1]), min(x_test_svm[:, 1])))
    print("Class weights: ", class_weight)
    
    if load:
        model = DNN_Model(load = load, save_model_name = save_model_name, 
                          num_set = num_set, detection = detection)
    else:
        input_dims = x_train_lstm.shape[2]
        time_steps = x_train_lstm.shape[1]
        svm_dims = x_train_svm.shape[1]

        model = LSTM_Model(svm_dims = svm_dims, input_shape = input_dims, num_classes = 2, detection = detection, num_set = num_set, save_model_name = save_model_name, time_steps = time_steps)
    
    # 训练模型
    print('-------------------------------- Start --------------------------------')
    model.train(x_train_lstm = x_train_lstm, x_train_svm = x_train_svm, y_train = y_train, 
                x_val_lstm = x_test_lstm, x_val_svm = x_test_svm, y_val = y_val,
                n_epochs = Config.EPOCHS, class_weight = class_weight)

    model.save_model()
    print('---------------------------------- End ----------------------------------')


def Test(load_model_name: str, num_set: str, detection: str, dataset: str, csvfile = None, load = False):
    
    process = PROCESS(num_set = num_set)
    x_train_svm, x_train_lstm, y_train, x_test_svm, x_test_lstm, y_test = process.dataset_split(detection = detection, dataset = dataset)
    
    x_train_lstm = np.array(pad_sequences(x_train_lstm, padding = "post", dtype = "float32"), dtype = np.float32) 
    x_test_lstm = np.array(pad_sequences(x_test_lstm, padding = "post", dtype = "float32"), dtype = np.float32)
    
    x_train_svm = x_train_svm[:, Config.SVM_FEATURE_SET]
    x_test_svm = x_test_svm[:, Config.SVM_FEATURE_SET]
   
    # one-hot
    y_train = k_utils.to_categorical(y_train)
    y_val = k_utils.to_categorical(y_test)

    print('------------------------------- Data Check --------------------------')
    print("SVM features: ", x_train_svm.shape, x_test_svm.shape)
    print("LSTM featrues: ", x_train_lstm.shape, x_test_lstm.shape)
    print("Sequential mean: ", np.mean(x_train_lstm[0, :, 1]), "Sequential std: ", np.std(x_train_lstm[0, :, 1]))
    print("SVM Train max: {} min: {}  Test max: {} min: {}".format(max(x_train_svm[:, 1]), min(x_train_svm[:, 1]), 
                                                                   max(x_test_svm[:, 1]), min(x_test_svm[:, 1])))
    model = DNN_Model(load = load, save_model_name = load_model_name, 
                      num_set = num_set, detection = detection)
    
    x_test_dict = {"svm_features": x_test_svm, "lstm_features": x_test_lstm}
    y_test, probs, predictions = model.evaluate(x_test_dict, y_test, num_set)
    print('---------------------------------- End ----------------------------------')

    return y_test, probs, predictions 


