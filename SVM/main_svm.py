import os
import sys
import csv
import warnings
import numpy as np

import matplotlib.pyplot as plt
from inspect import signature
from dataset_process_svm import SVM_PROCESS

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score, average_precision_score, precision_recall_curve

warnings.filterwarnings('ignore')

model_path = "./model/svm_model1"
model_name = os.path.join(model_path, "svm_model_A")

def Acc(y_test, y_pred, classification, display):
    
    if classification == 2:
        total_count = len(y_test)
        pos_count = list(y_test).count(1)
        neg_count = list(y_test).count(-1)

        acc = 0
        pos_acc = 0
        neg_acc = 0
        for i in range(total_count):
            if y_test[i] == y_pred[i]:
                acc += 1
            if y_test[i] == 1 and y_pred[i] == 1:
                pos_acc += 1
            if y_test[i] == -1 and y_pred[i] == -1:
                neg_acc += 1
        
        if display:
            print("Accuracy: {:.3f} Pos Accuracy: {:.3f} Neg Accuracy: {:.3f}\n".format(
                acc/total_count, pos_acc/pos_count, neg_acc/neg_count))
        
        f1 = f1_score(y_true = y_test, y_pred = y_pred, pos_label = 1)

        return neg_acc/neg_count, pos_acc/pos_count, acc/total_count, pos_acc, pos_count, neg_acc, neg_count, f1
    
    elif classification == 3:
        total_count = len(y_test)
        pos_count = list(y_test).count(2)
        neu_count = list(y_test).count(1)
        neg_count = list(y_test).count(0)

        acc = 0
        pos_acc = 0
        neu_acc = 0
        neg_acc = 0
        for i in range(total_count):
            if y_test[i] == y_pred[i]:
                acc += 1
            if y_test[i] == 2 and y_pred[i] == 2:
                pos_acc += 1
            if y_test[i] == 1 and y_pred[i] == 1:
                neu_acc += 1
            if y_test[i] == 0 and y_pred[i] == 0:
                neg_acc += 1
        
        if display:
            print("Accuracy: {:.3f} Pos Accuracy: {:.3f} Neg Accuracy: {:.3f} Neu Accuracy: {:.3f} \n".format(
                acc/total_count, pos_acc/pos_count, neg_acc/neg_count, neu_acc/neu_count))

def GridSearch(x_train, y_train, x_test, y_test, classification, metrics):

    param_kernel = ["rbf"]
    param_C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    param_gamma = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    param_degree = [2,3]
    class_weights = compute_class_weight(class_weight = "balanced", classes = np.unique(y_train), y = y_train)

    if len(class_weights) == 2:
        class_weight = {-1: class_weights[0], 1: class_weights[1]}
    else:
        class_weight = {0: class_weights[0], 1: class_weights[1], 2: class_weights[2]}

    mean_acc_best = 0
    f1_best = 0
    neg_acc_best = 0
    pos_acc_best = 0
    ap_best = 0
    
    cache_size = 3000

    params_best = {'C': 1, 'gamma': 0.001, 'kernel': 'rbf', "degree": 2, 'class_weight': class_weight, 'cache_size': cache_size}
    
    l = 2
    for i in param_kernel:
        for j in param_C:
            for k in param_gamma:
                # for l in param_degree:
                    params = {"kernel": i, "C": j, "gamma": k, "degree": l, "class_weight": class_weight, 'cache_size': cache_size}
                    clf = SVC(**params)
                    clf.fit(x_train, y_train)
                    y_pred = clf.predict(x_test)
                    mean_acc = clf.score(x_test, y_test)
                    neg_acc, pos_acc, _, _, _, _, _, f1 = Acc(y_test, y_pred, 2, 0)

                    if metrics == "accuracy":
                        if mean_acc > mean_acc_best:
                            mean_acc_best = mean_acc
                            params_best = params
                    
                    elif metrics == "f1":
                        f1 = f1_score(y_true = y_test, y_pred = y_pred, average = "macro")
                        if f1 > f1_best:
                            f1_best = f1
                            params_best = params
                    
                    elif metrics == "AP":                    
                        y_score = clf.decision_function(x_test)
                        ap = average_precision_score(y_test, y_score, None)
                        if ap > ap_best:
                            ap_best = ap
                            params_best = params
                    
                    elif metrics == "neg_acc":

                        if neg_acc > neg_acc_best and mean_acc > 0.5:
                            neg_acc_best = neg_acc
                            params_best = params
                    
                    elif metrics == "pos_acc":
                        
                        if pos_acc > pos_acc_best and mean_acc > 0.5:
                            pos_acc_best = pos_acc
                            params_best = params
                    #print("kernel: {}  C: {}  gamma: {}  Acc:{:.3f}  Neg Acc: {:.3f}  Pos Acc: {:.3f}".format(params["kernel"], params["C"], params["gamma"], mean_acc, neg_acc, pos_acc))
                    
    for i in [1,10, 100]:
        params = {"kernel": "linear", "C": i, "class_weight": class_weight}
        clf = SVC(**params)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        mean_acc = clf.score(x_test, y_test)
        neg_acc, pos_acc, _, _, _, _, _, f1 = Acc(y_test, y_pred, 2, 0)

        if metrics == "accuracy":
    
            if mean_acc > mean_acc_best:
                mean_acc_best = mean_acc
                params_best = params
        
        elif metrics == "f1":
            f1 = f1_score(y_true = y_test, y_pred = y_pred, average = "macro")
            if f1 > f1_best:
                f1_best = f1
                params_best = params
        
        elif metrics == "AP":                    
            y_score = clf.decision_function(x_test)
            ap = average_precision_score(y_test, y_score, None)
            if ap > ap_best:
                ap_best = ap
                params_best = params
        
        elif metrics == "neg_acc":
            
            if neg_acc > neg_acc_best and mean_acc > 0.5:
                neg_acc_best = neg_acc
                params_best = params
        
        elif metrics == "pos_acc":

            if pos_acc > pos_acc_best and mean_acc > 0.5:
                pos_acc_best = pos_acc
                params_best = params
    # print("kernel: {}  C: {}  f1_score: {:.3f}".format("linear", i, f1)) 

    return params_best

def PR_curve(y_test, y_score, average_precision):
    
    precision, recall, _ = precision_recall_curve(y_test, y_score, pos_label = 1)

    step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})

    plt.step(recall, precision, color = 'b', alpha = 0.2, where = 'post')
    plt.fill_between(recall, precision, alpha = 0.2, color = 'b', **step_kwargs)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("2-class Precision-Recall curve AP={0:0.2f}".format(average_precision))
    plt.savefig("Figure/pr.png")

def csvwrite():
 
    feature_dir = dataset_process_svm.feature_dir
    csv_dir =  dataset_process_svm.csv_dir
    test_dir = dataset_process_svm.test_dir
    csv_test = dataset_process_svm.csv_test

    csvfile1 = csv.reader(open(dataset_process_svm.csv_test))
    header = next(csvfile1)

    csvfile2 = csv.writer(open(os.path.join(model_path, "Test.csv"), "w"))
    header.insert(-2, "PRED")
    header.insert(-1, "RESULT")
    csvfile2.writerow(header)
   
    for index, row in enumerate(csvfile1):
        for i in os.listdir(test_dir):
            if i.split(".c")[0] == row[0] + "_" + row[3] + "_" + row[4]:

                x_data = np.array(dataset_process_svm.pickle_load(os.path.join(test_dir, i))[:-1], dtype = np.float)
                x_data = x_data.reshape(1, -1)
                y_data = np.int(dataset_process_svm.pickle_load(os.path.join(test_dir, i))[-1])
                y_pred = clf.predict(x_data)
                row.insert(27, y_pred[0])
                if y_pred[0] == y_data:
                    row.insert(28, "True")
                else:
                    row.insert(28, "False")
                csvfile2.writerow(row)

def svm_run(num_set = "A", scaling = 0, search_flag = 0, param = None, detection = None, metrics = None, display = 0):
    
    svm_process = SVM_PROCESS(num_set = num_set, scaling = scaling)
    x_train, y_train, x_test, y_test = svm_process.dataset_split(detection = detection)
    
    """
    index_PD = 0
    index_QD = 2

    for index, row in x_train:
        x_train[index, index_PD] = x_train[index, index_PD] * 100
        x_train[index, index_QD] = x_train[index, index_QD] * 100
    
    for index, row in x_test:
        x_test[index, index_PD] = x_test[index, index_PD] * 100
        x_test[index, index_QD] = x_test[index, index_QD] * 100
    """

    x_train = x_train[:, [6, 10]]
    x_test = x_test[:, [6, 10]]
    
    if len(x_train[1, :]) == 1:
        x_train = x_train.reshape(-1, 1)
        x_test = x_test.reshape(-1, 1)

    if search_flag:
        
        params_best = GridSearch(x_train, y_train, x_test, y_test, 2, metrics)
        
        if params_best['kernel'] == "rbf":
            print("\nkernel: {}  gamma: {}  C: {}  class_weight: {}\n".format(
                params_best["kernel"], params_best["gamma"], params_best["C"], params_best["class_weight"]))
        elif params_best['kernel'] == "linear":
            print("\nkernel: {}  C: {}  class_weight: {}\n".format(
                params_best["kernel"], params_best["C"], params_best["class_weight"]))
        
        return params_best

    else:
        
        class_weights = compute_class_weight(class_weight = "balanced", classes = np.unique(y_train), y = y_train)

        if len(class_weights) == 2:
            class_weight = {-1: class_weights[0], 1: class_weights[1]}
        else:
            class_weight = {0: class_weights[0], 1: class_weights[1], 2: class_weights[2]}

        param["class_weight"] = class_weight

        clf = SVC(**param)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        neg_acc, pos_acc, mean_acc, pos_true, pos_count, neg_true, neg_count, f1 = Acc(y_test, y_pred, classification = 2, display = display)
        
        # y_score = clf.decision_function(x_test)
        # ap = average_precision_score(y_test, y_score, None)

        return neg_acc, pos_acc, mean_acc, pos_true, pos_count, neg_true, neg_count, f1
     
