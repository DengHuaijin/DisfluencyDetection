import os
import sys
import csv
import pickle
import numpy as np
    
wavdir = "/mnt/work/WorkSpace/DataSet/CSJ-6th/CSJ-6th/WAV/all"
wavdir2 = "./WAV/wav/"
feature_csv = "./WAV/feature_set3/feature_csv"
feature_pickle = "./WAV/feature_set3/feature_pickle"
config_file = "./opensmileConfig/prosodyAcf.conf"

def pickle_dump(obj, filename):
    f = open(filename, "wb")
    pickle.dump(obj, f)
    f.close()

def wav_gen():
    
    csvfile = csv.reader(open("/mnt/work/WorkSpace/DataSet/CSJ-6th/svm/3class/impression_avg_svm-012.csv", "r"))
    head = next(csvfile)

    for index, row in enumerate(csvfile):
        filename = row[0] + "_" + row[3] + "_" + row[4]

        start_time = np.float(row[3])
        end_time = np.float(row[4])
        inputfile = os.path.join(wavdir, row[0] + ".wav")
        outputfile = os.path.join(wavdir2, filename + ".wav")
        cmd = "sox " + inputfile + " " + outputfile + " " + "trim " + str(start_time) + " " + str(end_time - start_time)
        print(outputfile)

        os.system(cmd)

def feature_gen():

    for index, i in enumerate(os.listdir(wavdir2)):
        csvfile = i.split(".w")[0] + ".csv"
        cmd = "SMILExtract -C " + config_file + " -I " + os.path.join(wavdir2, i) + " -O " + os.path.join(feature_csv, csvfile)
        os.system(cmd)

        csvread = csv.reader(open(os.path.join(feature_csv, csvfile), "r"), delimiter = ";")
        header = next(csvread)
        
        frames = []
        for index, row in enumerate(csvread):
            frames.append(row[2:])

        frames = np.round(np.array(frames, dtype = np.float32), decimals = 3)
        pickle_dump(frames, os.path.join(feature_pickle, i.split(".w")[0] + ".cpickle"))

feature_gen()


