import os
import sys
import h5py
import pickle
import pydot_ng
import tensorflow
import numpy as np

from Config import Config
from utils import plotCurve
from tensorflow.keras import backend as k
from tensorflow.keras.backend import squeeze
from tensorflow.keras import Input, optimizers, Model, callbacks, regularizers
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Bidirectional, PReLU
from tensorflow.keras.layers import Dropout, concatenate, Masking, LeakyReLU, Activation, dot
from tensorflow.keras.models import load_model as keras_load_model

class DNN_Model():
    
    def __init__(self, num_classes = None, num_set = None, 
            detection = None, save_model_name = None, load = None, **params):
        
        self.detection = detection
        self.num_set = num_set
        self.save_model_name = save_model_name 
        self.num_classes = num_classes
        
        if load:
            self.model = self.load_model(load_model_name = save_model_name, detection = detection, num_set = num_set)
        
        else:
            self.model_name = save_model_name
            self.model = self.make_model()
            self.model.compile(loss = 'binary_crossentropy', 
                               optimizer = optimizers.Adam(lr = Config.LR), 
                               metrics = ['accuracy'])

        print(self.model.summary(), file = sys.stderr)

        tensorflow.keras.utils.plot_model(self.model, 'Fig/multi_input_model.png', show_shapes = True)
    def save_model(self):
        
        h5_save_path = os.path.join("Models", self.detection, self.model_name + "_" + self.num_set + "_last.h5")
        self.model.save(h5_save_path)
    
    def load_model(self, load_model_name: str, detection: str, num_set: str):
 
        model_path = os.path.join("Models", detection, load_model_name + "_" + num_set + "_last.h5") 
        model = keras_load_model(model_path)

        return model

    def train(self, x_train_lstm, x_train_svm, y_train, 
              x_val_lstm = None, x_val_svm = None, y_val = None, 
              x_train_u = None, x_val_u = None,
              n_epochs = 20, class_weight = None):
         
        logdir = os.path.join("tensorboard_log", self.detection, self.save_model_name, self.num_set)
        cmd = "mkdir -p " + logdir
        os.system(cmd)
        cmd = "rm -r " + logdir + "/"
        os.system(cmd)
        tensorboard_callback = callbacks.TensorBoard(log_dir = logdir)
        
        h5_save_path = os.path.join("Models", self.detection, self.model_name + "_" + self.num_set + ".h5")

        acc = []
        loss = []
        val_acc = []
        val_loss = []
        
        x_train_dict = {'lstm_features': x_train_lstm, 'svm_features': x_train_svm, 'attention_params': x_train_u}
        x_val_dict = {'lstm_features':x_val_lstm, 'svm_features': x_val_svm, 'attention_params': x_val_u}
        
        train_set = tensorflow.data.Dataset.from_tensor_slices((x_train_dict, y_train)).shuffle(buffer_size = Config.BUFFER_SIZE).batch(32).prefetch(tensorflow.data.experimental.AUTOTUNE)
        val_set = tensorflow.data.Dataset.from_tensor_slices((x_val_dict, y_val)).shuffle(buffer_size = Config.BUFFER_SIZE).batch(32).prefetch(tensorflow.data.experimental.AUTOTUNE)

        model_cp = callbacks.ModelCheckpoint(filepath = h5_save_path,
                                             monitor = "val_loss",
                                             verbose = 0,
                                             save_best_only = True,
                                             save_weights_only = False,
                                             mode = "min",
                                             period = 10)
        """
        history = self.model.fit(x = x_train_dict, 
                                 y = y_train, 
                                 shuffle = 1,
                                 validation_data = (x_val_dict, y_val),
                                 batch_size = Config.BATCH_SIZE, 
                                 epochs = Config.EPOCHS,
                                 class_weight = class_weight,
                                 callbacks = [model_cp, tensorboard_callback],
                                 use_multiprocessing = False)
        """
        history = self.model.fit(x = train_set,
                                 validation_data = val_set,
                                 epochs = Config.EPOCHS,
                                 verbose = 1,
                                 class_weight = class_weight,
                                 callbacks = [model_cp, tensorboard_callback],
                                 use_multiprocessing = False)
 
        # 训练集上的损失值和准确率
        loss = history.history['loss']
        val_loss = history.history["val_loss"]
        
        figfile = os.path.join("Fig", self.detection, self.model_name + "_" + self.num_set + ".png")
        pickle.dump(loss, open(figfile.split(".p")[0] + "_loss.cpickle", "wb"))
        pickle.dump(val_loss, open(figfile.split(".p")[0] + "_val_loss.cpickle", "wb"))
        plotCurve(loss, val_loss, 'Model Loss', 'loss', figfile)
        
    def evaluate(self, x_test, y_test, num_set):
       
        probs = self.model.predict(x_test)
        predictions = np.argmax(probs, axis = 1)

        print(y_test)
        print(predictions)
       
        return y_test, probs, predictions 
    
    def make_model(self):
        raise NotImplementedError()


class LSTM_Model(DNN_Model):

    def __init__(self, svm_dims, input_shape, time_steps, **params):
        params['name'] = 'LSTM'
        self.svm_dims = svm_dims
        self.time_steps = time_steps
        self.input_shape = input_shape
        super(LSTM_Model, self).__init__(**params)

    def make_model(self):
        
        with k.name_scope("SVM_features"):

            svm_features = Input(shape = (self.svm_dims,), name = "svm_features")
            svm_input = Dense(128, activation = "tanh", name = "svm_dense")(svm_features)
            # svm_input = LeakyReLU()(svm_input) 
        
        with k.name_scope("LSTM_features"):
            
            lstm_features = Input(shape = (None, self.input_shape), name = "lstm_features")
            lstm_mask = Masking(mask_value = Config.MASKING_VALUE, input_shape = (self.time_steps, self.input_shape))(lstm_features)
            lstm_output = LSTM(Config.LSTM_UNITS, return_sequences = True,  name = "lstm_sequence")(lstm_mask)
            lstm_output_last = LSTM(Config.LSTM_UNITS, return_sequences = False, name = "lstm_last_output")(lstm_mask)
        
        with k.name_scope("AttentionLayer_1"):
            
            lstm_dense = Dense(128, activation = None, use_bias = False, name = "lstm_dense")(lstm_output) # [B, M, 128]
            attention_weights1 = dot([lstm_output_last, lstm_dense], name = "attention_weights1", axes = -1) # [B, 1, M]
            attention_weights2 = Activation("softmax", name = "attention_weights2")(attention_weights1)
            lstm_attention = dot([attention_weights2, lstm_output], name = "lstm_attention", axes = 1)
        
        """
        with k.name_scope("AttentionLayer_2"):
            # Attention layer 2 - attention params
            input_attention = Input(shape = (Config.ATTENTION_UNITS, ), name = "attention_params")
            u = Dense(Config.ATTENTION_UNITS, activation = "softmax", name = "attention_u")(input_attention)
            alpha = dot([u, lstm_output], axes = -1)
            alpha = Activation("softmax", name = "attention_weights")(alpha)
            # weighted pool
            lstm_attention = dot([alpha, lstm_output], name = "attention_output", axes = 1)
        """
        with k.name_scope("Concatenate"):
            x = concatenate([lstm_attention, svm_input])
            x_dense = Dense(128, activation = "tanh")(x)
            # x_dense = LeakyReLU()(x_dense)
            dense_2 = Dense(128, activation = "tanh")(x_dense)
            batchnorm2 = BatchNormalization()(dense_2)
            dropout = Dropout(rate = 0.3, name = "dropout")(batchnorm2) 
            
        pred = Dense(self.num_classes, activation = "softmax", name = "output")(dropout)
        self.model = Model(inputs = [svm_features, lstm_features], outputs = [pred])
        
        return self.model

