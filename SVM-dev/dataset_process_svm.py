import os
import sys
import csv
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

class SVM_PROCESS():

    def __init__(self, num_set, scaling):
        
        self.scaling = scaling
        self.num_set = num_set
        feature_dir = "D:\\nlpworkspace\\slack\\Speech\\nachos\\3class\\all_features_dev"
        csv_dir =  "D:\\nlpworkspace\\slack\\Speech\\taco\\CSV\\CV012-5"
        
        self.train_dir = os.path.join(feature_dir, "feature_" + num_set + "\\train")
        self.dev_dir = os.path.join(feature_dir, "feature_" + num_set + "\\dev")
        self.test_dir = os.path.join(feature_dir, "test")
        self.retrain_dir = os.path.join(feature_dir, "retrain")
        
        self.csv_train = os.path.join(csv_dir, "impression_avg_svm-012_train_" + num_set + ".csv")
        self.csv_dev = os.path.join(csv_dir, "impression_avg_svm-012_dev_" + num_set + ".csv")
        self.csv_retrain = os.path.join(csv_dir, "impression_avg_svm-012_train.csv")
        self.csv_test = os.path.join(csv_dir, "impression_avg_svm-012_test.csv")
    
    def pickle_dump(self, obj, filename):
        f = open(filename, "wb")
        pickle.dump(obj, f)
        f.close()

    def pickle_load(self, filename):
        f = open(filename, "rb")
        result = pickle.load(f)
        f.close()

        return result

    def feature_process(self):
        # 14     15     16      17      18         19         20          21          22   23        24    25    26
        # 0      1      2       3       4          5          6           7           8    9         10    11    12
        # D_WORD F_WORD D2_WORD F2_WORD ORI_D_WORD ORI_F_WORD ORI_D2_WORD ORI_F2_WORD WORD ORI_SPEED SPEED A05   LABEL
        # DF1:P DF2:Q DF3:PF+QD DF4:PD+QF DF5:SPEED+QD DF6:SPEED+QF
        
        print("data set: {}".format(self.num_set))

        train_dir = self.train_dir
        dev_dir = self.dev_dir
        retrain_dir = self.retrain_dir
        test_dir = self.test_dir

        csv_train = self.csv_train
        csv_dev = self.csv_dev
        csv_retrain = self.csv_retrain
        csv_test = self.csv_test

        cmd = "del " + train_dir + "/*"
        os.system(cmd)
        cmd = "del " + dev_dir + "/*"
        os.system(cmd)
        cmd = "del " + retrain_dir + "/*"
        os.system(cmd)
        cmd = "del " + test_dir + "/*"
        os.system(cmd)

        csvfile1 = csv.reader(open(csv_train, "r"))
        csvfile2 = csv.reader(open(csv_dev, "r"))
        csvfile3 = csv.reader(open(csv_retrain, "r"))
        csvfile4 = csv.reader(open(csv_test, "r"))
        header = next(csvfile1)
        header = next(csvfile2)
        header = next(csvfile3)
        header = next(csvfile4)
        
        feature = []
        for index, row in enumerate(csvfile1):
            feature = np.array(row[14:])
            self.pickle_dump(feature, os.path.join(train_dir, row[0] + "_" + row[3] + "_" + row[4] + ".cpickle"))
        
        print("train file: {}".format(index + 1))

        for index, row in enumerate(csvfile2):
            feature = np.array(row[14:])
            self.pickle_dump(feature, os.path.join(dev_dir, row[0] + "_" + row[3] + "_" + row[4] + ".cpickle"))

        print("dev file: {}".format(index + 1))

        for index, row in enumerate(csvfile3):
            feature = np.array(row[14:])
            self.pickle_dump(feature, os.path.join(retrain_dir, row[0] + "_" + row[3] + "_" + row[4] + ".cpickle"))
        
        print("retrain file: {}".format(index + 1))

        for index, row in enumerate(csvfile4):
            feature = np.array(row[14:])
            self.pickle_dump(feature, os.path.join(test_dir, row[0] + "_" + row[3] + "_" + row[4] + ".cpickle"))

        print("test file: {}\n".format(index + 1))


    def scale_process(self, x_train, x_test):
        
        # print("scaling ...")

        scaler = preprocessing.MinMaxScaler(copy = True, feature_range = (0, 1))

        x_train = np.array(x_train, dtype = np.float)
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)

        x_test = np.array(x_test, dtype = np.float)
        scaler.fit(x_test)
        x_test = scaler.transform(x_test)
        
        return x_train, x_test

    def dataset_split(self, detection = None, data_set = None):
      
        num_set = self.num_set
        if data_set == "dev":
            train_dir = self.train_dir
            test_dir = self.dev_dir
        elif data_set == "test":
            train_dir = self.retrain_dir
            test_dir = self.test_dir

        # print("data set: {}".format(num_set))

        x_train = []
        y_train = []
        x_test = []
        y_test = []

        train_set = os.listdir(train_dir)
        np.random.shuffle(train_set)
        test_set = os.listdir(test_dir)
        np.random.shuffle(test_set)

        for i in train_set:
            feature = self.pickle_load(os.path.join(train_dir, i))
            x_train.append(feature[:-1])
            y_train.append(feature[-1])

        for i in test_set:
            feature = self.pickle_load(os.path.join(test_dir, i))
            x_test.append(feature[:-1])
            y_test.append(feature[-1])
        
        if  self.scaling:
            x_train, x_test = self.scale_process(x_train, x_test)

        x_train = np.array(x_train, dtype = np.float)
        y_train = np.array(y_train, dtype = np.int32)
        x_test = np.array(x_test, dtype = np.float)
        y_test = np.array(y_test, dtype = np.int32)

        if detection == "positive":

            for index, label in enumerate(y_train):
                if label == 2:
                    y_train[index] = 1
                else:
                    y_train[index] = -1

            for index, label in enumerate(y_test):
                if label == 2:
                    y_test[index] = 1
                else:
                    y_test[index] = -1

        elif detection == "negative":
            
            for index, label in enumerate(y_train):
                if label == 0:
                    y_train[index] = -1
                else:
                    y_train[index] = 1

            for index, label in enumerate(y_test):
                if label == 0:
                    y_test[index] = -1
                else:
                    y_test[index] = 1

        return x_train, y_train, x_test, y_test

