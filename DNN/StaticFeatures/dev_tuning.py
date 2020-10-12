from __future__ import division, absolute_import, print_function

import sys,os
import csv
import pickle
import argparse
import numpy as np
from collections import OrderedDict

import tensorflow as tf
from tensorflow.keras import backend as K
from data_process import PROCESS
from Config import Config
from dnn_model import DiluteLayer, CustomF1
from sklearn.metrics import roc_curve

parser = argparse.ArgumentParser()
parser.add_argument("--params", required = True)
parser.add_argument("--detection", default = "negative")
parser.add_argument("--gpu", default = "0")
parser.add_argument("--nosvm", default = 0)
parser.add_argument("--dataset", default = "dev")
parser.add_argument("--plot", default = 0)
parser.add_argument("--benchmark", default = "default")
parser.add_argument("--tag", default = "A05")
parser.add_argument("--epoch", default = 0)
args = parser.parse_args()

params = args.params
detection = args.detection
gpu = args.gpu
nosvm = int(args.nosvm)
dataset = args.dataset
plot = int(args.plot)
tag = args.tag
benchmark = args.benchmark
EPOCH = int(args.epoch)

os.environ["CUDA_VISIBLE_DEVICES"] = gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
K.set_session(sess)

all_set = ["A", "B", "C", "D", "E"]

if __name__ == "__main__":

    if not os.path.exists(os.path.join("dev_tuning", tag, detection, params)):
        os.mkdir(os.path.join("dev_tuning", tag, detection, params))
    
    if not plot:
        csvfile = csv.writer(open("dev_tuning/" + tag + "/" + detection + "/" + params + ".csv", "w", newline = ""))
        head = ["EPOCH", "recall 0-10", "recall 10-20", "recall 20-30", "recall 30-40", "f-measure", "recall"]
        csvfile.writerow(head)
    
        cmd = "cp " + os.path.join("Models", detection, "model-dev_tuning*") + " " + os.path.join("dev_tuning", tag, detection, params)
        # os.system(cmd)
    
    if plot:
        if benchmark not in ["recall0-10", "recall", "fmeasure", "none"]:
            raise ValueError("recall0-10, recall, fmeasure", "none")
        
        csvfile2 = csv.writer(open("dev_tuning/" + tag + "/" + detection + "/" + "plot" + "/" + params + "_" + benchmark + "_" + str(EPOCH) + ".csv", "w", newline = ""))
        head2 = ["recall", "precision"]
        csvfile2.writerow(head2)
        
        csvfile3 = csv.writer(open("dev_tuning/" + tag + "/" + detection + "/" + "plot" + "/" + "roc/" + params + "_" + benchmark + "_" + str(EPOCH) + ".csv", "w", newline = ""))
        head3 = ["fpr", "tpr", "threshold"]
        csvfile3.writerow(head3)

    # for epoch in range(10,410,10):
    for epoch in [EPOCH]:
        pred_sum = []
        prob_sum = []
        labels_sum = []
        
        for num_set in all_set:

            # model_name = os.path.join("Models", detection, "model-dev_tuning_" + num_set + "_" + str(epoch) + ".h5")
            model_name = os.path.join("dev_tuning", tag, detection, params, "model-dev_tuning_" + num_set + "_" + str(epoch) + ".h5")
            model = tf.keras.models.load_model(model_name, custom_objects = {"CustomF1": CustomF1}, compile = False)

            process = PROCESS(num_set = num_set, tag = tag)

            x_train_svm, x_train_lstm, y_train, x_test_svm, x_test_lstm, y_test = process.dataset_split(detection = detection, dataset = dataset)
            x_test_svm = x_test_svm[:, Config.SVM_FEATURE_SET]
            if nosvm:
                # print("nosvm")
                x_test_svm = np.zeros(shape = x_test_svm.shape)
            x_test = {"svm_features": x_test_svm, "lstm_features": x_test_lstm}

            probs = model.predict(x_test)
            predictions = np.argmax(probs, axis = 1)
            
            pred = [max(i, 0) for i in predictions]
            label = [max(i, 0) for i in y_test]
            # print(num_set, " label:", label)
            # print(num_set, " pred:", pred)
            if detection == "negative":
                pred = [i^1 for i in pred]
                label = [i^1 for i in label]
                prob = probs[:, 0]
            elif detection == "positive":
                prob = probs[:, 1]
            else:
                raise ValueError
            
            pred_sum.extend(pred)
            prob_sum.extend(prob)
            labels_sum.extend(label)
        
        prob_rank = sorted(set(prob_sum))[::-1]
        # print(prob_sum)

        index_label = []
        threshold_dict = OrderedDict()
        total_labels = labels_sum.count(1)
        for index, i in enumerate(pred_sum):
            if i == 1:
                index_label.append(index)
        for threshold in prob_rank:
            threshold_dict[threshold] = [0,0,0,0]
            for index in index_label:
                if prob_sum[index] >= threshold:
                    threshold_dict[threshold][1] += 1
                    if labels_sum[index] == 1:
                        threshold_dict[threshold][0] += 1
            
            threshold_dict[threshold][2] = np.round(threshold_dict[threshold][0] / total_labels, decimals = 5)
            if  threshold_dict[threshold][1] != 0:
                threshold_dict[threshold][3] = np.round(threshold_dict[threshold][0] / threshold_dict[threshold][1], decimals = 5)
            else:
                threshold_dict[threshold][3] = 0
        
        precision1 = []
        precision2 = []
        precision3 = []
        precision4 = []
        for key in threshold_dict.keys():
            if plot:
                row = [str(threshold_dict[key][2]), str(threshold_dict[key][3])]
                print(row)
                csvfile2.writerow(row)
            #print(key, threshold_dict[key])
            if 0.1 >= threshold_dict[key][2] >= 0:
                precision1.append(threshold_dict[key][3])
            elif 0.2 >= threshold_dict[key][2] > 0.1:
                precision2.append(threshold_dict[key][3])
            elif 0.3 >= threshold_dict[key][2] > 0.2:
                precision3.append(threshold_dict[key][3])
            elif 0.4 >= threshold_dict[key][2] > 0.3:
                precision4.append(threshold_dict[key][3])
            else:
                continue
        
        if not plot:
            if precision1 == []: 
                recall_0_10 = 0 
            else: recall_0_10 = np.round(np.mean(precision1), decimals = 5)
            if precision2  == []: 
                recall_10_20 = 0 
            else: recall_10_20 = np.round(np.mean(precision2), decimals = 5)
            if precision3 == []: 
                recall_20_30 = 0 
            else: recall_20_30 = np.round(np.mean(precision3), decimals = 5)
            if precision4 == []: 
                recall_30_40 = 0 
            else: recall_30_40 = np.round(np.mean(precision4), decimals = 5)
            try:
                f_measure = np.round(2 / (1 / threshold_dict[key][2] + 1 / threshold_dict[key][3]), decimals = 5)
            except:
                f_measure = 0
            recall = np.round(threshold_dict[key][2], decimals = 5)

            line = [str(epoch), str(recall_0_10), str(recall_10_20), str(recall_20_30), str(recall_30_40), str(f_measure), str(recall)]
            print(line)
            csvfile.writerow(line)
            # sys.exit(0)

        if plot:
            fpr, tpr, thresholds = roc_curve(y_true = labels_sum, y_score = prob_sum, pos_label = 1, drop_intermediate = False)
            eps = 1
            eer = 0
            for index, value in enumerate(fpr):
                row = [str(fpr[index]), str(tpr[index]), str(thresholds[index])]
                csvfile3.writerow(row)
                if abs(1 - fpr[index] - tpr[index]) < eps:
                    eps = abs(1 - fpr[index] - tpr[index]) 
                    eer = fpr[index]
            print("EER = {}".format(eer))


        

