import os
import sys
import h5py
import pickle
import pydot_ng
import tensorflow as tf
import numpy as np

from utils import plotCurve
from Config import Config
from tensorflow.keras import layers, backend as k
from tensorflow.keras.backend import squeeze
from tensorflow.keras import Input, optimizers, Model, callbacks, regularizers
from tensorflow.keras.layers import Lambda, Conv2D, MaxPooling2D, Reshape, multiply
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Bidirectional, PReLU
from tensorflow.keras.layers import Dropout, concatenate, Masking, LeakyReLU, Activation, dot
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.utils.generic_utils import to_list

class DNN_Model():
    
    def __init__(self, num_classes = None, num_set = None, 
            detection = None, save_model_name = None, load = None, **params):
        
        self.detection = detection
        self.num_set = num_set
        self.save_model_name = save_model_name 
        self.num_classes = num_classes
        self.load = load

        if self.detection == "negative":
            class_id = 0
        elif self.detection == "positive":
            class_id = 1
        else:
            raise ValueError()

        if load:
            self.model = self.load_model(load_model_name = save_model_name, detection = detection, num_set = num_set)
            self.model_name = save_model_name
            self.model.compile(loss = 'binary_crossentropy', 
                               optimizer = optimizers.Adam(lr = Config.LR), 
                               metrics = ['accuracy', CustomF1(name = "f1", class_id = class_id), Precision(class_id = class_id), Recall(class_id = class_id)])
        else:
            self.model_name = save_model_name
            self.model = self.make_model()
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(Config.LR, decay_steps = 10000, decay_rate = 0.96, staircase = True)
            self.model.compile(loss = 'binary_crossentropy', 
                               optimizer = optimizers.Adam(learning_rate = Config.LR), 
                               metrics = ['accuracy', CustomF1(name = "f1", class_id = class_id), Precision(class_id = class_id), Recall(class_id = class_id)])
        print(self.model.summary(), file = sys.stderr)

        tf.keras.utils.plot_model(self.model, 'Fig/multi_input_model.png', show_shapes = True)
    
    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * 0.1

    def save_model(self):
        
        h5_save_path = os.path.join("Models", self.detection, self.model_name + "_" + self.num_set + "_last.h5")
        self.model.save(h5_save_path)
    
    def load_model(self, load_model_name: str, detection: str, num_set: str):
 
        model_path = os.path.join("Models", detection, load_model_name + "_" + num_set + "_last.h5")
        
        model = keras_load_model(model_path, custom_objects = {"DiluteLayer": DiluteLayer, "CustomF1": CustomF1}, compile = False)

        return model

    def train(self, x_train_lstm, x_train_svm, y_train, 
              x_val_lstm = None, x_val_svm = None, y_val = None, 
              x_train_u = None, x_val_u = None,
              n_epochs = 20, class_weight = None):
        
        x_train_dict = {'svm_features': x_train_svm, 'lstm_features': x_train_lstm}
        x_val_dict = {'svm_features': x_val_svm, 'lstm_features': x_val_lstm}
        
        train_set = tf.data.Dataset.from_tensor_slices((x_train_dict, y_train)).shuffle(buffer_size = Config.BUFFER_SIZE).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
        val_set = tf.data.Dataset.from_tensor_slices((x_val_dict, y_val)).shuffle(buffer_size = Config.BUFFER_SIZE).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

        logdir = os.path.join("tensorboard_log", self.detection, self.save_model_name, self.num_set)
        if not os.path.exists(logdir):
            cmd = "mkdir -p " + logdir
            os.system(cmd)
        if not self.load  and os.path.exists(logdir):
            cmd = "rm -r " + logdir + "/*"
            os.system(cmd)
        if self.load and not os.path.exists(logdir):
            raise ValueError()

        tensorboard_callback = callbacks.TensorBoard(log_dir = logdir, write_graph = False) 
        lr_callback = callbacks.LearningRateScheduler(self.scheduler)

        h5_save_path = os.path.join("Models", self.detection, self.model_name + "_" + self.num_set + "_{epoch:02d}.h5")

        acc = []
        loss = []
        val_acc = []
        val_loss = []
        
        model_cp = callbacks.ModelCheckpoint(filepath = h5_save_path,
                                             monitor = "val_loss",
                                             verbose = 0,
                                             save_best_only = False,
                                             save_weights_only = False,
                                             mode = "min",
                                             period = 10)
        
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
        predictions = []
        #for i in probs:
        #    if i[0] >= 0.50:
        #        predictions.append(0)
        #    else:
        #        predictions.append(1)
        predictions = np.argmax(probs, axis = 1)

        print(y_test)
        print(predictions)
       
        return y_test, probs, predictions 
    
    def make_model(self):
        raise NotImplementedError()

class CustomF1(tf.keras.metrics.Metric):

    def __init__(self, thresholds = None, top_k = None, class_id = None, name = None, dtype = None):
        
        super(CustomF1, self).__init__(name = name, dtype = dtype)
        self.init_thresholds = thresholds
        self.top_k = top_k
        self.class_id = class_id

        default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
        self.thresholds = metrics_utils.parse_init_thresholds(
                thresholds, default_threshold = default_threshold)
        self.true_positives = self.add_weight("true_positives", shape = (len(self.thresholds),), initializer = init_ops.zeros_initializer)
        self.false_negatives = self.add_weight("fasle_negatives", shape = (len(self.thresholds),), initializer = init_ops.zeros_initializer)
        self.false_positives = self.add_weight("fasle_positives", shape = (len(self.thresholds),), initializer = init_ops.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight = None):
       
        op = metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives,
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives
            },
            y_true,
            y_pred,
            thresholds = self.thresholds,
            top_k = self.top_k,
            class_id = self.class_id,
            sample_weight = sample_weight)

    def result(self):
        
        recall = math_ops.div_no_nan(self.true_positives, self.true_positives + self.false_negatives)
        precision = math_ops.div_no_nan(self.true_positives, self.true_positives + self.false_positives)
        f1 = 2/(1/recall[0] + 1/precision[0])

        return f1
    
    def reset_states(self):
       
       num_thresholds = len(to_list(self.thresholds))
       k.batch_set_value(
               [(v, np.zeros((num_thresholds,))) for v in self.variables])

    def get_config(self):
        config = {
                "thresholds": self.init_thresholds,
                "top_k": self.top_k,
                "class_id": self.class_id
                }
        base_config = super(CustomF1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class CustomPR(tf.keras.metrics.Metric):

    def __init__(self, detection = None, name = "CustomPR", **kwargs):
        super(CustomPR, self).__init__(name = name, **kwargs)
        self.detection = detection
        self.NegTrue = self.add_weight(name = "negtrue", initializer = "zeros")

    def update_state(self, y_true, y_pred, sample_weight = None):
        
        NegTrue = k.cast(k.equal(y_true[:, 0] + y_pred[:, 0],0), k.floatx())
        PosTrue = k.cast(k.equal(y_true[:, 1] + y_pred[:, 1],2), k.floatx())
        
        Neg_Precision = k.sum(NegTrue) / k.sum(y_pred[:, 0])
        Pos_Precision = k.sum(PosTrue) / k.sum(y_pred[:, 1])
        Neg_Recall = k.sum(NegTrue) / k.sum(y_true[:, 0])
        Pos_Recall = k.sum(PosTrue) / k.sum(y_true[:, 1])
        Neg_f1 = 0.5*(1/Neg_Precision + 1/Neg_Recall)
        Pos_f1 = 0.5*(1/Pos_Precision + 1/Pos_Recall)
        
        self.NegTrue.assign_add(k.sum(PosTrue))

    def result(self):
        return self.NegTrue

class DiluteLayer(layers.Layer):

    def __init__(self, output_dim = 27, **params):
        
        super(DiluteLayer, self).__init__(**params)
        self.output_dim = output_dim

    def build(self, input_shape):

        self.w = self.add_weight(shape = (1, self.output_dim), 
                                 initializer = "uniform",
                                 dtype = 'float32', 
                                 trainable = True)

        self.b = self.add_weight(shape = (1, self.output_dim), 
                                 initializer = "zeros",
                                 dtype = 'float32', 
                                 trainable = True)
    
    def call(self, inputs, **params):
        return inputs * self.w + self.b

    def get_config(self):
        config = super(DiluteLayer, self).get_config()
        config.update({
            "output_dim":self.output_dim
            })
        return config

diluteLayer = DiluteLayer(27, name = "dense_diluteLayer")

class LSTM_Model(DNN_Model):

    def __init__(self, svm_dims = None, input_shape = None, time_steps = None, **params):
        params['name'] = 'LSTM'
        self.svm_dims = svm_dims
        self.time_steps = time_steps
        self.input_shape = input_shape

        super(LSTM_Model, self).__init__(**params)

    def make_model(self):
        
        with k.name_scope("SVM_features"):

            svm_features = Input(shape = (self.svm_dims,), name = "svm_features")
            svm_input = Dense(128, activation = "tanh", name = "svm_dense1", kernel_regularizer = None)(svm_features)
            """
            # svm_input = Dense(128, activation = "tanh", name = "svm_dense2", kernel_regularizer = None)(svm_input)
            self_attention = Dense(7, activation = "sigmoid", kernel_regularizer = None, name = "svm_sigmoid")(svm_features)
            self_attention = Activation("softmax")(self_attention)
            svm_weighted  =multiply([svm_input, self_attention])
            # svm_input = LeakyReLU()(svm_input)
            # svm_input = Dropout(rate = Config.DROP)(svm_input)
            """
        with k.name_scope("FUNC_features"):

            func_features = Input(shape = (self.input_shape,), name = "lstm_features")
            func_input = Dense(128, activation = "tanh", name = "func_dense1", kernel_regularizer = None)(func_features)
            """ 
            self_attention = Dense(6373, activation = "sigmoid", kernel_regularizer = None, name = "func_sigmoid")(func_features)
            self_attention = Activation("softmax")(self_attention)
            func_weighted = multiply([func_input, self_attention])
            # func_input = Dense(128, activation = "tanh", name = "func_dense2", kernel_regularizer = None)(func_input)
            # func_input = LeakyReLU()(func_input)
            # func_input = Dropout(rate = Config.DROP)(func_input)"
            """
        with k.name_scope("Concatenate"):
            x = concatenate([svm_input, func_input])
            
            # dense_dilute = diluteLayer(x, name = "dilute_1")
            # dense_dilute = Activation("sigmoid")(dense_dilute) # 1~-1 scaling
            dense_0 = Dense(256, activation = "tanh", kernel_regularizer = regularizers.l2(0.001))(x)
            self_attention = Dense(256, activation = "sigmoid", kernel_regularizer = regularizers.l2(0.001), name = "sigmoid")(x)
            self_attention = Activation("softmax")(self_attention)
            
            weighted = multiply([dense_0, self_attention])
            # weighted = Dropout(rate = )(weighted)
            
            dense_1 = Dense(128, activation = "tanh", kernel_regularizer = None)(x)
            # dense_1 = LeakyReLU()(dense_1)
            # BN_1 = BatchNormalization()(dense_1)
            # dense_1 = Dropout(rate = Config.DROP)(dense_1)
            
            dense_2 = Dense(128, activation = "tanh", kernel_regularizer = None)(dense_1)
            # dense_2 = LeakyReLU()(dense_2)
            
            BN_2 = BatchNormalization()(dense_2)
            dropout = Dropout(rate = Config.DROP)(BN_2)
            
            dense_3 = Dense(128, activation = "tanh", kernel_regularizer = regularizers.l2(0.001))(dense_2)
            # dense_3 = LeakyReLU()(dense_3)
            # dense_3 = Dropout(rate = Config.DROP)(dense_3)
            BN_3 = BatchNormalization()(dense_3)
            #dropout = Dropout(rate = Config.DROP)(BN_3)

            dense_4 = Dense(128, activation = "tanh", kernel_regularizer = regularizers.l2(0.001))(dense_3)
            # desne_4 = LeakyReLU()(dense_4)
            BN_4 = BatchNormalization()(dense_4)
            # dropout = Dropout(rate = Config.DROP)(BN_4)

            dense_5 = Dense(128, activation = "tanh", kernel_regularizer = regularizers.l2(0.001))(dense_4)
            # BN_5 = BatchNormalization()(dense_5)
            # dropout = Dropout(rate = Config.DROP)(BN_5)
            
        pred = Dense(self.num_classes, activation = "softmax", name = "output")(dropout)
        self.model = Model(inputs = [svm_features, func_features], outputs = [pred])
        
        return self.model

