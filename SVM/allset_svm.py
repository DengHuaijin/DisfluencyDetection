import os
import sys
import csv
import main_svm
import argparse
import numpy as np
from dataset_process_svm import SVM_PROCESS
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight

all_set = ["A", "B", "C", "D", "E"]

parser = argparse.ArgumentParser()
parser.add_argument("--feature", default = 0)
parser.add_argument("--gridsearch", default = 0)
parser.add_argument("--acc", default = 0)
parser.add_argument("--detection")
parser.add_argument("--metrics")
parser.add_argument("--scaling", default = 0)
args = parser.parse_args()

detection = args.detection
metrics = args.metrics
scaling = args.scaling

def feature_generation():

    for num_set in all_set:

        svm_process = SVM_PROCESS(num_set, scaling = 0)
        svm_process.feature_process()

def params_generation():
    
    svm_process = SVM_PROCESS("A", scaling = scaling)
    params_best = []

    for num_set in all_set:
        
        params_best.append(main_svm.svm_run(num_set = num_set, scaling = scaling, search_flag = 1, detection = detection, metrics = metrics))
    
    svm_process.pickle_dump(params_best, "./params_best.pickle")

def accuracy_generation():
   
    acc_temp = csv.writer(open("acc_temp.csv", "w", newline = ""))
    row = ["", "Negative Acc", "Positive Acc", "Mean Acc", "F1"]
    acc_temp.writerow(row)

    svm_process = SVM_PROCESS("A", scaling = scaling)
    params_best = svm_process.pickle_load("./params_best.pickle")
   
    for param in params_best:
        print(param)

        neg_list = []
        pos_list = []
        acc_list = []
        f1_list = []

        TP_list = []
        FN_list = []
        TN_list = []
        FP_list = []
        
        for num_set in all_set:

            neg_acc, pos_acc, mean_acc, _, _, _, _, f1= main_svm.svm_run(num_set = num_set, scaling = scaling, search_flag = 0, param = param, detection = detection, metrics = metrics, display = 1)
            
            neg_list.append(neg_acc)
            pos_list.append(pos_acc)
            acc_list.append(mean_acc)
            f1_list.append(f1)

            neg_acc = str(np.round(neg_acc, decimals = 3)*100) + "%"
            pos_acc = str(np.round(pos_acc, decimals = 3)*100) + "%"
            mean_acc = str(np.round(mean_acc, decimals = 3)*100) + "%"
            f1 = str(np.round(f1, decimals = 3)*100) + "%"

            row = [num_set, neg_acc, pos_acc, mean_acc, f1]
            acc_temp.writerow(row)

        neg_mean = str(np.round(np.mean(neg_list), decimals = 4)*100) + "%"
        pos_mean = str(np.round(np.mean(pos_list), decimals = 4)*100) + "%"
        acc_mean = str(np.round(np.mean(acc_list), decimals = 4)*100) + "%"
        f1_mean = str(np.round(np.mean(f1_list), decimals = 4)*100) + "%"

        row = ["Average", neg_mean, pos_mean, acc_mean, f1_mean]
        acc_temp.writerow(row)
        acc_temp.writerow("")

        for num_set in all_set:
            
            _, _, _, pos_true, pos_count, neg_true, neg_count, f1 = main_svm.svm_run(num_set = num_set, scaling = scaling, search_flag = 0, param = param, detection = detection, metrics = metrics, display = 0)

            TP = pos_true
            FP = neg_count - neg_true
            FN = pos_count - pos_true
            TN = neg_true

            TP_list.append(TP)
            FN_list.append(FN)
            FP_list.append(FP)
            TN_list.append(TP)

            row = [num_set, "Label Positive", "Label Negative"]
            acc_temp.writerow(row)
            row = ["Predicted Positive", str(TP), str(FP)]
            acc_temp.writerow(row)
            row = ["Predicted Negative", str(FN), str(TN)]
            acc_temp.writerow(row)
            
        acc_temp.writerow("")
        row = [num_set, "Label Positive", "Label Negative"]
        acc_temp.writerow(row)
        row = ["Predicted Positive", str(np.mean(TP_list)), str(np.mean(FP_list))]
        acc_temp.writerow(row)
        row = ["Predicted Negative", str(np.mean(FN_list)), str(np.mean(TN_list))]
        acc_temp.writerow(row)
        acc_temp.writerow("")

if args.feature:
    feature_generation()
elif args.gridsearch:
    params_generation()
elif args.acc:
    accuracy_generation()
