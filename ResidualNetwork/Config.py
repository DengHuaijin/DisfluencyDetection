# 参数配置
import os

class Config:

    SVM_FEATURE_PATH = "/mnt/work/DataSet/CSJ-6th/svm/3class/all_features_dev/"
    SVM_CSV_PATH = "/mnt/work/DataSet/CSJ-6th/svm/3class/cv7/"
    SPEC_FEATURE_PATH = os.path.join("WAV", "spec1", "feature_pickle")
    WAV_PATH = os.path.join("WAV", "wav")    
    SVM_FEATURE_SET = [4,5,6,7,10]
