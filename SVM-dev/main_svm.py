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
from sklearn.metrics import classification_report, f1_score, average_precision_score, precision_score

warnings.filterwarnings('ignore')

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
        
        f1_neg = f1_score(y_true = y_test, y_pred = y_pred, pos_label = -1)
        f1_pos = f1_score(y_true = y_test, y_pred = y_pred, pos_label = 1)
        precision_neg = precision_score(y_true = y_test, y_pred = y_pred, pos_label = -1)
        precision_pos = precision_score(y_true = y_test, y_pred = y_pred, pos_label = 1)

        if display:
            print("Mean Accuracy: {:.3f} Pos Accuracy: {:.3f} Neg Accuracy: {:.3f}\n".format(
                acc/total_count, pos_acc/pos_count, neg_acc/neg_count))

        return neg_acc/neg_count, pos_acc/pos_count, acc/total_count, pos_acc, pos_count, neg_acc, neg_count, f1_neg, precision_neg, f1_pos, precision_pos
    
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

def CrossValidation(param = "", detection = None):
	
	neg_list = []
	pos_list = []
	acc_list = []
	f1_list = []
	precision_list = []
	
	for num_set in ["A", "B", "C", "D", "E"]:

		svm_process = SVM_PROCESS(num_set = num_set, scaling = 1)
		x_train, y_train, x_test, y_test = svm_process.dataset_split(detection = detection, data_set = "dev")
		 
		if feature_set == "":
			feature_set = [13 ,14]
		else:
			feature_set = feature_set
		
		x_train = x_train[:, feature_set]
		x_test = x_test[:, feature_set]
		
		if len(x_train[1, :]) == 1:
			x_train = x_train.reshape(-1, 1)
			x_test = x_test.reshape(-1, 1)
		
		class_weights = compute_class_weight(class_weight = "balanced", classes = np.unique(y_train), y = y_train)

		if len(class_weights) == 2:
			class_weight = {-1: class_weights[0], 1: class_weights[1]}
		else:
			class_weight = {0: class_weights[0], 1: class_weights[1], 2: class_weights[2]}
		
		params["class_weight"] = class_weight
		  
		clf = SVC(**params)
		clf.fit(x_train, y_train)
		y_pred = clf.predict(x_test)
		neg_acc, pos_acc, mean_acc, _, _, _, _, f1_neg, precision_neg, f1_pos, precision_pos = Acc(y_test, y_pred, 2, 0)

		neg_list.append(neg_acc)
		pos_list.append(pos_acc)
		acc_list.append(mean_acc)

		if detection == "negative":
			f1_list.append(f1_neg)
			precision_list.append(precision_neg)
		elif detection == "positive":
			f1_list.append(f1_pos)
			precision_list.append(precision_pos)

	neg_mean = np.mean(np.array(neg_list))
	pos_mean = np.mean(np.array(pos_list))
	acc_mean = np.mean(np.array(acc_list))
	f1_mean = np.mean(np.array(f1_list))
	precision_mean = np.mean(np.array(precision_list))
	
	return neg_mean, pos_mean, acc_mean, f1_mean, precision_mean

def GridSearch(detection = None, classification = 2, metrics = "f1", feature_set = ""):

    param_kernel = ["rbf"]
    param_C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    param_gamma = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    f1_best = 0
    neg_best = 0
    pos_best = 0
    precision_best = 0
    
    cache_size = 3000

    params_best = {'C': 1, 'gamma': 0.001, 'kernel': 'rbf', 'class_weight': None, 'cache_size': cache_size}
	
    recall_ths = 0.3
    acc_ths = 0.5
    
    print("rbf...")
    for i in param_kernel:
        for j in param_C:
            for k in param_gamma:
                # for l in param_degree:
                    params = {"kernel": i, "C": j, "gamma": k, "class_weight": None, 'cache_size': cache_size}
                    
                    neg_list = []
                    pos_list = []
                    acc_list = []
                    f1_list = []
                    precision_list = []
                    
                    for num_set in ["A", "B", "C", "D", "E"]:

                        svm_process = SVM_PROCESS(num_set = num_set, scaling = 1)
                        x_train, y_train, x_test, y_test = svm_process.dataset_split(detection = detection, data_set = "dev")
                         
                        if feature_set == "":
                            feature_set = [13 ,14]
                        else:
                            feature_set = feature_set
                        
                        x_train = x_train[:, feature_set]
                        x_test = x_test[:, feature_set]
                        
                        if len(x_train[1, :]) == 1:
                            x_train = x_train.reshape(-1, 1)
                            x_test = x_test.reshape(-1, 1)
                        
                        class_weights = compute_class_weight(class_weight = "balanced", classes = np.unique(y_train), y = y_train)

                        if len(class_weights) == 2:
                            class_weight = {-1: class_weights[0], 1: class_weights[1]}
                        else:
                            class_weight = {0: class_weights[0], 1: class_weights[1], 2: class_weights[2]}
                        
                        params["class_weight"] = class_weight
                          
                        clf = SVC(**params)
                        clf.fit(x_train, y_train)
                        y_pred = clf.predict(x_test)
                        neg_acc, pos_acc, mean_acc, _, _, _, _, f1_neg, precision_neg, f1_pos, precision_pos = Acc(y_test, y_pred, 2, 0)

                        neg_list.append(neg_acc)
                        pos_list.append(pos_acc)
                        acc_list.append(mean_acc)

                        if detection == "negative":
                            f1_list.append(f1_neg)
                            precision_list.append(precision_neg)
                        elif detection == "positive":
                            f1_list.append(f1_pos)
                            precision_list.append(precision_pos)

                    neg_mean = np.mean(np.array(neg_list))
                    pos_mean = np.mean(np.array(pos_list))
                    acc_mean = np.mean(np.array(acc_list))
                    f1_mean = np.mean(np.array(f1_list))
                    precision_mean = np.mean(np.array(precision_list))
                    
                    if metrics == "f1":
                        if f1_mean > f1_best:
                            f1_best = f1_mean
                            params_best = params
                    
                    elif metrics == "precision":
                        
                        if detection == "negative":
                            if precision_mean > precision_best and neg_mean > recall_ths:
                                precision_best = precision_mean
                                params_best = params
                        
                        elif detection == "positive":
                            if precision_mean > precision_best and pos_mean > recall_ths:
                                precision_best = precision_mean
                                params_best = params
                    
                    elif metrics == "neg_acc":
                        if neg_mean > neg_best and acc_mean > acc_ths:
                            neg_best = neg_mean
                            params_best = params
                    
                    elif metrics == "pos_acc":
                        if pos_mean > pos_best and acc_mean > acc_ths:
                            pos_best = pos_mean
                            params_best = params
                    #print("kernel: {}  C: {}  gamma: {}  Acc:{:.3f}  Neg Acc:{:.3f}  Neg f1:{:.3f}  Pos Acc:{:.3f}".format(params["kernel"], params["C"], params["gamma"], acc_mean, neg_mean, f1_mean, pos_mean))
    
    print("linear...")
    for i in [1, 10, 100]:
        params = {"kernel": "linear", "C": i, "class_weight": class_weight}
        clf = SVC(**params)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        mean_acc = clf.score(x_test, y_test)
        neg_acc, pos_acc, _, _, _, _, _, f1_neg, precision_neg, f1_pos, precision_pos = Acc(y_test, y_pred, 2, 0)
                    
        neg_list = []
        pos_list = []
        acc_list = []
        f1_list = []
        precision_list = []

        for num_set in ["A", "B", "C", "D", "E"]:

            svm_process = SVM_PROCESS(num_set = num_set, scaling = 1)
            x_train, y_train, x_test, y_test = svm_process.dataset_split(detection = detection, data_set = "dev")
            
            if feature_set == "":
                feature_set = [13 ,14]
            else:
                feature_set = feature_set
                if len(feature_set) == 1:
                    feature_set = [feature_set]

            x_train = x_train[:, feature_set]
            x_test = x_test[:, feature_set]
            
            if len(x_train[1, :]) == 1:
                x_train = x_train.reshape(-1, 1)
                x_test = x_test.reshape(-1, 1)
            
            class_weights = compute_class_weight(class_weight = "balanced", classes = np.unique(y_train), y = y_train)

            if len(class_weights) == 2:
                class_weight = {-1: class_weights[0], 1: class_weights[1]}
            else:
                class_weight = {0: class_weights[0], 1: class_weights[1], 2: class_weights[2]}
            
            params["class_weight"] = class_weight

            clf = SVC(**params)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            neg_acc, pos_acc, mean_acc, _, _, _, _, f1_neg, precision_neg, f1_pos, precision_pos = Acc(y_test, y_pred, 2, 0)

            neg_list.append(neg_acc)
            pos_list.append(pos_acc)
            acc_list.append(mean_acc)
            
            if detection == "negative":
                f1_list.append(f1_neg)
                precision_list.append(precision_neg)
            elif detection == "positive":
                f1_list.append(f1_pos)
                precision_list.append(precision_pos)

        neg_mean = np.mean(np.array(neg_list))
        pos_mean = np.mean(np.array(pos_list))
        acc_mean = np.mean(np.array(acc_list))
        f1_mean = np.mean(np.array(f1_list))
        precision_mean = np.mean(np.array(precision_list))
                         
        if detection == "negative":
            f1_list.append(f1_neg)
            precision_list.append(precision_neg)
        elif detection == "positive":
            f1_list.append(f1_pos)
            precision_list.append(precision_pos)
       
        if metrics == "f1":
            if f1_mean > f1_best:
                f1_best = f1_mean
                params_best = params
        
        elif metrics == "precision":
            if detection == "negative":
                if precision_mean > precision_best and neg_mean > recall_ths:
                    precision_best = precision_mean
                    params_best = params
            elif detection == "positive":
                if precision_mean > precision_best and pos_mean > recall_ths:
                    precision_best = precision_mean
                    params_best = params
        
        elif metrics == "neg_acc":
            if neg_mean > neg_best and acc_mean > acc_ths:
                neg_best = neg_mean
                params_best = params
        
        elif metrics == "pos_acc":
            if pos_mean > pos_best and acc_mean > acc_ths:
                pos_best = pos_mean
                params_best = params

    return params_best

def svm_run(num_set = "A", scaling = 0, search_flag = 0, param = None, detection = None, metrics = None, display = 0, feature_set = ""):
    
    if search_flag:
        
        params_best = GridSearch(detection, 2, metrics, feature_set)
        
        if params_best['kernel'] == "rbf":
            print("\nkernel: {}  gamma: {}  C: {}  class_weight: {}\n".format(
                params_best["kernel"], params_best["gamma"], params_best["C"], params_best["class_weight"]))
        elif params_best['kernel'] == "linear":
            print("\nkernel: {}  C: {}  class_weight: {}\n".format(
                params_best["kernel"], params_best["C"], params_best["class_weight"]))
        
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
            feature_set = [13 ,14]
        else:
            feature_set = feature_set
        
        x_train = x_train[:, feature_set]
        x_test = x_test[:, feature_set]

        clf = SVC(**param)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        neg_acc, pos_acc, mean_acc, pos_true, pos_count, neg_true, neg_count, f1_neg, precision_neg, f1_pos, precision_pos = Acc(y_test, y_pred, classification = 2, display = display)

        return neg_acc, pos_acc, mean_acc, pos_true, pos_count, neg_true, neg_count, f1_neg, precision_neg, f1_pos, precision_pos
     