import os
import sys
import csv
import warnings
import numpy as np

import matplotlib.pyplot as plt
from inspect import signature
from sklearn.metrics import classification_report, f1_score, average_precision_score, precision_score

warnings.filterwarnings('ignore')

def Acc(y_test, y_pred, classification, display):
    
    if classification == 2:
        total_count = len(y_test)
        pos_count = list(y_test).count(1)
        neg_count = list(y_test).count(0)

        acc = 0
        pos_acc = 0
        neg_acc = 0
        for i in range(total_count):
            if y_test[i] == y_pred[i]:
                acc += 1
            if y_test[i] == 1 and y_pred[i] == 1:
                pos_acc += 1
            if y_test[i] == 0 and y_pred[i] == 0:
                neg_acc += 1

        if neg_acc == 0:
            print("0 neg_acc neg_acc = 1")
            neg_acc = 1
        if pos_acc == 0:
            print("0 pos_acc pos_acc = 1")
            pos_acc = 1
        
        f1_neg = f1_score(y_true = y_test, y_pred = y_pred, pos_label = 0)
        f1_pos = f1_score(y_true = y_test, y_pred = y_pred, pos_label = 1)
        precision_neg = precision_score(y_true = y_test, y_pred = y_pred, pos_label = 0)
        precision_pos = precision_score(y_true = y_test, y_pred = y_pred, pos_label = 1)
        
        if display:
            print("Accuracy: {:.3f}  Pos Recall: {:.3f}  Pos Precision: {:.3f}  Pos f1: {:.3f}  Neg Recall: {:.3f}  Neg Precision: {:.3f}  Neg f1: {:.3f}\n".format(acc/total_count, pos_acc/pos_count, precision_pos, f1_pos, neg_acc/neg_count, precision_neg,  f1_neg))
 
        return neg_acc/neg_count, pos_acc/pos_count, acc/total_count, pos_acc, pos_count, neg_acc, neg_count, f1_neg, precision_neg, f1_pos, precision_pos

def svm_run(num_set = "", scaling = 0, search_flag = 0, param = None, detection = None, metrics = None, display = 0, feature_set = ""):
    
    if search_flag:
        
        params_best = GridSearch(num_set = num_set, detection = detection, classification = 2, metrics = metrics, feature_set = feature_set)
        
        if params_best['kernel'] == "rbf":
            print("\nkernel: {}  gamma: {}  C: {}  class_weight: {}".format(
                params_best["kernel"], params_best["gamma"], params_best["C"], params_best["class_weight"]))
        elif params_best['kernel'] == "linear":
            print("\nkernel: {}  gamma: {}  C: {}  class_weight: {}".format(
                params_best["kernel"], params_best['gamma'], params_best["C"], params_best["class_weight"]))
        
        return params_best

    else:
        
        svm_process = SVM_PROCESS(num_set = num_set, scaling = 1)
        x_train, y_train, x_test, y_test = svm_process.dataset_split(detection = detection, data_set = "test")
        
        class_weights = compute_class_weight(class_weight = "balanced", classes = np.unique(y_train), y = y_train)

        if len(class_weights) == 2:
            class_weight = {-1: class_weights[0], 1: class_weights[1]}
        else:
            class_weight = {0: class_weights[0], 1: class_weights[1], 2: class_weights[2]}

        param["class_weight"] = class_weight
        
        if feature_set == "":
            feature_set = [4, 10, 13 ,14]
        else:
            feature_set = feature_set
        
        x_train = x_train[:, feature_set]
        x_test = x_test[:, feature_set]

        clf = SVC(**param)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        neg_acc, pos_acc, mean_acc, pos_true, pos_count, neg_true, neg_count, f1_neg, precision_neg, f1_pos, precision_pos = Acc(y_test, y_pred, classification = 2, display = display)
        

        return neg_acc, pos_acc, mean_acc, pos_true, pos_count, neg_true, neg_count, f1_neg, precision_neg, f1_pos, precision_pos
     

