# 参数配置
import os

class Config:

    SVM_FEATURE_PATH = "/mnt/work/WorkSpace/DataSet/CSJ-6th/svm/3class/all_features_dev/"
    SVM_CSV_PATH = "/mnt/work/DataSet/CSJ-6th/svm/3class/cv7/"
    ACOUSTIC_FEATURE_PATH = os.path.join("WAV", "feature_set3", "feature_pickle")
   
    ATTENTION_UNITS = 128
    ATTENTION_INIT = 1.0 / ATTENTION_UNITS

    LSTM_UNITS = 128
    MASKING_VALUE = -1

    SVM_FEATURE_SET = [5, 7, 10, 13, 14]

    EPOCHS = 300
    LR = 0.00001
    BATCH_SIZE = 32
    BUFFER_SIZE = 128

