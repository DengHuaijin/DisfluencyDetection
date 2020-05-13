import os
import sys
import csv
import diss_classify
import argparse
import pickle
import numpy as np
import tensorflow as tf

from main_svm import Acc
from data_process import PROCESS
from tensorflow.keras import backend as K

parser = argparse.ArgumentParser()
parser.add_argument("--train", default = 0)
parser.add_argument("--test", default = 0)
parser.add_argument("--load", default = 0)
# python main_allset.py --set A B C D E
parser.add_argument("--set", nargs = "*", required = True)
parser.add_argument("--model", default = None, required = True)
parser.add_argument("--detection", default = None, required = True)
parser.add_argument("--gpu", default = "0", required = True)
parser.add_argument("--dataset", default = "dev")
args = parser.parse_args()

detection = args.detection
load = args.load
gpu = args.gpu
all_set = args.set
model_name = args.model
dataset = args.dataset

### GPU ###
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.Session(config = config)
K.set_session(sess)
##########

def Train():

    save_model_name = model_name

    for num_set in all_set:
        
        diss_classify.Train(save_model_name = save_model_name, num_set = num_set, detection = detection, dataset = "dev", load = load)
        

def Test():
        
    acc_test = csv.writer(open("Test/acc_Test.csv", "w", newline = ""))
    
    row = []
    if detection == "negative":
        row = ["", "Negative Recall", "Negative F1", "Negative Precision", "Anti-Negative Recall", "Anti-Negative F1", "Anti-Negative Precision", "Mean Acc"]
    elif detection == "positive":
        row = ["", "Anti-Positive Recall", "Anti-Positive F1", "Anti-Positive Precision", " Positive Recall", "Positive F1", "Positive Precision", "Mean Acc"]
    acc_test.writerow(row)
     
    label_list = []
    probs_list = []
    pred_list = []
    tmp = []

    neg_list = []
    pos_list = []
    acc_list = []
    f1_neg_list = []
    precision_neg_list = []
    f1_pos_list = []
    precision_pos_list = []

    for num_set in all_set:
        
        y_test, probs, predictions = diss_classify.Test(load_model_name = model_name, num_set = num_set, detection = detection, dataset = dataset, csvfile = acc_test, load = load)
        
        tmp = [label_list.append(i) for i in y_test]
        tmp = [probs_list.append(i) for i in probs]
        tmp = [pred_list.append(i) for i in predictions]
        
        pickle.dump(predictions, open("Test/" + detection + "/" + num_set + "_pred.cpickle", "wb"))
        pickle.dump(probs, open("Test/" + detection + "/" + num_set + "_prob.cpickle", "wb"))
        pickle.dump(y_test, open("Test/" + detection + "/" + num_set + "_label.cpickle", "wb"))

        neg_acc, pos_acc, mean_acc, pos_true, pos_count, neg_true, neg_count, f1_neg, precision_neg, f1_pos, precision_pos = Acc(y_test, predictions, 2, 1)
        neg_acc = str(np.round(neg_acc, decimals = 3)*100) + "%"
        pos_acc = str(np.round(pos_acc, decimals = 3)*100) + "%"
        mean_acc = str(np.round(mean_acc, decimals = 3)*100) + "%"
        f1_neg = str(np.round(f1_neg, decimals = 3)*100) + "%"
        precision_neg = str(np.round(precision_neg, decimals = 3)*100) + "%"
        f1_pos = str(np.round(f1_pos, decimals = 3)*100) + "%"
        precision_pos = str(np.round(precision_pos, decimals = 3)*100) + "%"

        row = [num_set, neg_acc, f1_neg, precision_neg, pos_acc, f1_pos, precision_pos, mean_acc]
        acc_test.writerow(row)
 
    pickle.dump(label_list, open("./Test/" + detection + "/" + "labels.cpickle", "wb"))
    pickle.dump(probs_list, open("./Test/" + detection + "/" + "probs.cpickle", "wb"))
    pickle.dump(pred_list, open("./Test/" + detection + "/" + "preds.cpickle", "wb"))

if __name__ == "__main__":
        
    if args.train:
        Train()

    elif args.test:
        Test()

