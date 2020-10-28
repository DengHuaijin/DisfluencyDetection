import os
import sys
import csv
import main_svm
import argparse
import numpy as np
import feature_comb
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
parser.add_argument("--result", default = 0)
parser.add_argument("--all_feature", default = 0)
args = parser.parse_args()

detection = args.detection
metrics = args.metrics
scaling = args.scaling

def feature_generation():

    for num_set in all_set:

        svm_process = SVM_PROCESS(num_set, scaling = 0)
        svm_process.feature_process()

def params_get(feature_set = "", num_set = "", detection = "", metrics = ""):
    
    feature_name = ""
    if feature_set == "":
        feature_set = [4, 10, 13, 14]
    for i in feature_set:
        feature_name += str(i) + "_"

    feature_name += ".cpickle"

    params_file = os.path.join("params", detection, num_set, metrics, feature_name)
   
    return params_file

def params_generation(feature_set = ""):
    
    for num_set in all_set:

        params_file = params_get(feature_set = feature_set, num_set = num_set, detection = detection, metrics = metrics)
        if os.path.exists(params_file):
            continue
        
        print("\ndata set {} GridSearch...".format(num_set)) 

        svm_process = SVM_PROCESS("A", scaling = scaling)
        
        params_best = []
        params_best = main_svm.svm_run(num_set = num_set, scaling = scaling, search_flag = 1, detection = detection, metrics = metrics, feature_set = feature_set)

        svm_process.pickle_dump(params_best, params_file)

def accuracy_generation(feature_set = ""):
   
    """
    acc_temp = csv.writer(open("acc_temp.csv", "w", newline = ""))
    
    if detection == "negative":
        row = ["", "Negative Acc", "Negative F1", "Negative Precision", "Anti-Neagtive Acc", "Anti-Negative F1", "Anti-Negative Precision", "Mean Acc"]
    elif detection == "positive":
        row = ["", "Anti-Positive Acc", "Anti-Positive F1", "Anti-Positive Precision", "Positive Acc", "Positive F1", "Positive Precision", "Mean Acc"]
    
    acc_temp.writerow(row)
    """
    f1_best = 0
    row_best = []
  
    neg_list = []
    pos_list = []
    acc_list = []
    f1_neg_list = []
    precision_neg_list = []
    f1_pos_list = []
    precision_pos_list = []

    TP_list = []
    FN_list = []
    TN_list = []
    FP_list = []
    
    for num_set in all_set:
       
        svm_process = SVM_PROCESS(num_set, scaling = 1)
        params_file = params_get(feature_set = feature_set, num_set = num_set, detection = detection, metrics = metrics)
        params_best = svm_process.pickle_load(params_file)
 
        neg_acc, pos_acc, mean_acc, _, _, _, _, f1_neg, precision_neg, f1_pos, precision_pos = main_svm.svm_run(num_set = num_set, scaling = scaling, search_flag = 0, param = params_best, detection = detection, metrics = metrics, display = 1, feature_set = feature_set)
        
        neg_list.append(neg_acc)
        pos_list.append(pos_acc)
        acc_list.append(mean_acc)
        f1_neg_list.append(f1_neg)
        precision_neg_list.append(precision_neg)
        f1_pos_list.append(f1_pos)
        precision_pos_list.append(precision_pos)

        neg_acc = str(np.round(neg_acc, decimals = 3)*100) + "%"
        pos_acc = str(np.round(pos_acc, decimals = 3)*100) + "%"
        mean_acc = str(np.round(mean_acc, decimals = 3)*100) + "%"
        f1_neg = str(np.round(f1_neg, decimals = 3)*100) + "%"
        precision_neg = str(np.round(precision_neg, decimals = 3)*100) + "%"
        f1_pos = str(np.round(f1_pos, decimals = 3)*100) + "%"
        precision_pos = str(np.round(precision_pos, decimals = 3)*100) + "%"

        # row = [num_set, neg_acc, f1_neg, precision_neg, pos_acc, f1_pos, precision_pos, mean_acc]
        # acc_temp.writerow(row)

    neg_mean = str(np.round(np.mean(neg_list), decimals = 4)*100) + "%"
    pos_mean = str(np.round(np.mean(pos_list), decimals = 4)*100) + "%"
    acc_mean = str(np.round(np.mean(acc_list), decimals = 4)*100) + "%"
    f1_neg_mean = str(np.round(np.mean(f1_neg_list), decimals = 4)*100) + "%"
    precision_neg_mean = str(np.round(np.mean(precision_neg_list), decimals = 4)*100) + "%"
    f1_pos_mean = str(np.round(np.mean(f1_pos_list), decimals = 4)*100) + "%"
    precision_pos_mean = str(np.round(np.mean(precision_pos_list), decimals = 4)*100) + "%"


    row = [neg_mean, f1_neg_mean, precision_neg_mean, pos_mean, f1_pos_mean, precision_pos_mean, acc_mean]
    # acc_temp.writerow(row)
    # acc_temp.writerow("")
    
    """
    for num_set in all_set:
        
        _, _, _, pos_true, pos_count, neg_true, neg_count, neg_f1, neg_precision, pos_f1, pos_precision = main_svm.svm_run(num_set = num_set, scaling = scaling, search_flag = 0, param = params_best, detection = detection, metrics = metrics, display = 0)

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
    row = ["Avg", "Label Positive", "Label Negative"]
    acc_temp.writerow(row)
    row = ["Predicted Positive", str(np.mean(TP_list)), str(np.mean(FP_list))]
    acc_temp.writerow(row)
    row = ["Predicted Negative", str(np.mean(FN_list)), str(np.mean(TN_list))]
    acc_temp.writerow(row)
    acc_temp.writerow("")
    """
    if args.all_feature:

        return row


def result_generation():
    
    svm_process = SVM_PROCESS("A", scaling = scaling)
    params_best = svm_process.pickle_load("./params_best.pickle")
    clf = SVC(**params_best[1])
    
    csvfile = csv.reader(open("/mnt/work/DataSet/CSJ-6th/svm/3class/cv2/impression_avg_svm-012_test_A.csv", "r"))
    header = next(csvfile)
    print(header[20], header[24])

    csvfile2 = csv.writer(open("./result.csv", "w", newline = ""))

    header.insert(31, "PRED")
    csvfile2.writerow(header)

    svm_process = SVM_PROCESS(num_set = "A", scaling = scaling)
    x_train, y_train, x_test, y_test = svm_process.dataset_split(detection = detection)
    
    feature_set = [6, 10]

    x_train = x_train[:, feature_set]
    x_test = x_test[:, feature_set]
    
    clf.fit(x_train, y_train)

    for index, row in enumerate(csvfile):
        
        if row[-1] == "0":
            row[-1] = "-1"
        else:
            row[-1] = "1"

        feature = [[np.float(row[20]), np.float(row[24])]]
        y_pred = clf.predict(feature)
        
        for i in [14,15,16,17,18,19,20,21]:
            row[i] = str(np.float(row[i])/100)

        row.insert(31, y_pred)
        csvfile2.writerow(row)
        

if args.feature:
    feature_generation()
elif args.gridsearch:
    params_generation()
elif args.acc:
    accuracy_generation()
elif args.result:
    result_generation()

elif args.all_feature:
   
    csvfile = detection + "_" + metrics + ".csv"
    all_result = csv.writer(open(csvfile, "w", newline = ""))
    
    if detection == "negative":
        row = ["", "Negative Acc", "Negative F1", "Negative Precision", "Anti-Neagtive Acc", "Anti-Negative F1", "Anti-Negative Precision", "Mean Acc"]
    elif detection == "positive":
        row = ["", "Anti-Positive Acc", "Anti-Positive F1", "Anti-Positive Precision", "Positive Acc", "Positive F1", "Positive Precision", "Mean Acc"]
 
    all_result.writerow(row)

    feature_dict = {"4": "1PD", "5":"1PF", "6": "1QD", "7": "1QF", "10": "SPEED", "13": "PS_WORD", "14": "SIL_AVG"}
    comb_set = feature_comb.combination_gen()

    for feature in comb_set:
        
        feature_set = []
        for i in feature:
            feature_set.append(feature_dict.get(str(i)))

        print(feature, feature_set)

        # params_generation(feature)
        row_best = accuracy_generation(feature)
        
        row_best.insert(0, feature_set)
        print(row_best, "\n")
        all_result.writerow(row_best)
