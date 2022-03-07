import os
import csv
import json
import logging
import warnings
import argparse

logger = logging.getLogger("Logger")
logger.setLevel(20)
sh = logging.StreamHandler()
logger.addHandler(sh)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_set", type = str, required = True)
    parser.add_argument("--detection", type = str, default = "negative")
    args = parser.parse_args()
    
    if args.detection not in ["negative", "positive"]:
        raise ValueError("detection should be negative or positive")
    if args.num_set not in ["A", "B", "C", "D", "E"]:
        raise ValueError("num_set should be A B C D E")

    csv_dir = os.path.join("/mnt", "work", "DataSet", "CSJ-6th", "svm", "3class", "cv7", args.num_set)
    wav_dir = os.path.join("WAV") 
    
    train_csv = os.path.join(csv_dir, "impression_avg_svm-012_train.csv")
    dev_csv = os.path.join(csv_dir, "impression_avg_svm-012_dev.csv")
    test_csv_ = os.path.join(csv_dir, "impression_avg_svm-012_test.csv")

    train_read = csv.reader(open(train_csv, "r"))
    dev_read = csv.reader(open(dev_csv, "r"))
    test_read = csv.reader(open(test_csv_, "r"))

    train_json = open(os.path.join("manifest", args.detection, args.num_set, "train_sp.json"), "w")
    # dev_json = open(os.path.join("manifest", args.detection, args.num_set, "dev.json"), "w")
    # test_json = open(os.path.join("manifest", args.detection, args.num_set, "test.json"), "w")
    
    wav_dict = {}
    for wavfile in os.listdir(wav_dir):
        wav_dict[wavfile] = 1
    
    for dataset in ["train"]:
        if dataset == "train": 
            csv_read = train_read
            json_file = train_json
        if dataset == "dev": 
            csv_read = dev_read
            json_file = dev_json
        if dataset == "test":
            csv_read = test_read
            json_file = test_json
        
        logger.log(20, "Processing {} dataset...".format(dataset))
        
        head = next(csv_read)
        for index, line in enumerate(csv_read):
            wavfile = line[0] + "_" + line[3] + "_" + line[4] + ".wav"
            
            if wav_dict.get(wavfile) == None:
                warnings.warn("wavfile {} not in {}".format(wavfile, wav_dir))
                continue
            
            duration = float(line[5])
            if args.detection == "negative":
                label = "negative" if line[-1] == "0" else "others"
            if args.detection == "positive":
                label = "positive" if line[-1] == "2" else "others"
            
            meta = {"audio_filepath": os.path.join(wav_dir, wavfile), "offset": 0, "duration": duration, "label": label}
            json.dump(meta, json_file)
            json_file.write("\n")
