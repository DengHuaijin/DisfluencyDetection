# 参数配置
import os

class Config:

    SVM_FEATURE_PATH = "/mnt/work/DataSet/CSJ-6th/svm/3class/all_features_dev/"
    SVM_CSV_PATH = "/mnt/work/DataSet/CSJ-6th/svm/3class/cv7/"
    ACOUSTIC_FEATURE_PATH = os.path.join("WAV", "feature_set5_func2", "feature_pickle")
   
    LSTM_UNITS = 128
    MASKING_VALUE = -1

    SVM_FEATURE_SET = [4,5,6,7,10,13,14]

    EPOCHS = 400
    LR = 0.0000001
    BATCH_SIZE = 32
    BUFFER_SIZE = 128
    DROP = 0.3
    
