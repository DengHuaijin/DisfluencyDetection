import os
import sys
import csv
import argparse
import diss_classify
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.utils as k_utils
from tensorflow.keras import backend
from data_process import PROCESS
from Config import Config
from dnn_model import DiluteLayer, CustomF1

parser = argparse.ArgumentParser()
parser.add_argument("--detection", default = None, required = True)
parser.add_argument("--model", default = None, required = True)
parser.add_argument("--set", default = None, required = True)
parser.add_argument("--gpu", default = "0")
args  =parser.parse_args()

detection = args.detection
model_name = args.model
num_set = args.set
gpu = args.gpu

os.environ["CUDA_VISIBLE_DEVICES"] = gpu

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
backend.set_session(sess)

def csv_write(func_out, df):
    
    func_out_str = []
    for i in func_out[0][0]:
        
        func_out_str.append(str(i))
        
    se = pd.Series(func_out_str, index = df.columns)
    df = df.append(se, ignore_index = True)

    return df
    
if __name__ == "__main__":

    model_name = os.path.join("Models", detection, model_name + "_" + num_set + "_last.h5")
    model = tf.keras.models.load_model(model_name, custom_objects = {"CustomF1": CustomF1}, compile = False)

    process = PROCESS(num_set = num_set)

    x_train_svm, x_train_lstm, y_train, x_test_svm, x_test_lstm, y_test, testname, y_tmp = process.dataset_split(detection = detection, dataset = "test")

    x_train_svm = x_train_svm[:, Config.SVM_FEATURE_SET]
    x_test_svm = x_test_svm[:, Config.SVM_FEATURE_SET]
   
    y_train = k_utils.to_categorical(y_train)
    y_val = k_utils.to_categorical(y_test)

    # columns = ["FP/Mr", "WF/Mr", "MrWF/Mr", "MrFP/Mr", "SpR", "Ps/Mr", "SilR", "VoiceProb_max", "VoiceProb_min", "VoiceProb_mean", "VoiceProb_var", "VoiceProb_stddev", "F0_max","F0_min", "F0_mean","F0_var", "F0_stddev","Energy_max", "Energy_min","Energy_mean", "Energy_var","Energy_stddev", "zcr_max","zcr_min", "zcr_mean","zcr_var", "zcr_stddev"]
    Dict = {"SPEECH":[], "START":[], "END":[], "Label":[], "PRED":[], "TF":[]}
    for index, feature in enumerate(y_test):
        
        x_test = {"svm_features": x_test_svm[index:index+1, :], "lstm_features": x_test_lstm[index:index+1, :]}
    
        prob = model.predict(x_test)
        prediction = np.argmax(prob)
        print("label: ", y_tmp[index], " prediction: ", prediction)
        
        speech = testname[index].split("_")[0]
        start = testname[index].split("_")[1]
        end = testname[index].split("_")[2].split(".cpickle")[0]

        Dict["SPEECH"].append(speech)
        Dict["START"].append(start)
        Dict["END"].append(end)
        Dict["Label"].append(y_tmp[index])
        Dict["PRED"].append(prediction)
        
        if y_tmp[index] == 0 and prediction == 0:
            Dict["TF"].append("True")
        elif (y_tmp[index] == 1 or y_tmp[index] == 2) and prediction == 1:
            Dict["TF"].append("True")
        else:
            Dict["TF"].append("False")
     
    df = pd.DataFrame(Dict, columns = ["SPEECH", "START", "END", "Label", "PRED", "TF"])
    df.to_csv(num_set + ".csv", index = False)        
    print(df)
        
            

