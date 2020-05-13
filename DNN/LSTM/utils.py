import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras.models import model_from_json
from sklearn.metrics import precision_recall_curve, average_precision_score
from tensorflow.keras.models import load_model as keras_load_model


def plotCurve_PR(labels = None, probs = None, figfile = None, detection = None):

    if detection == "negative":
        pos_label = 0
    else:
        pos_label = 1
    
    average_precision = average_precision_score(labels, probs, pos_label = pos_label)
    precision, recall, _ = precision_recall_curve(labels, probs, pos_label = pos_label)
    
    plt.figure()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.savefig(figfile)

'''
plotCurve(): 
    绘制损失值和准确率曲线

输入:
    train(list): 训练集损失值或准确率数组
    val(list): 测试集损失值或准确率数组
    title(str): 图像标题
    y_label(str): y 轴标题
'''
def plotCurve(train, val, title: str, y_label: str, figfile: str):
    
    plt.figure()
    plt.plot(train)
    plt.plot(val)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel('epoch')
    plt.legend(['train', 'Validation'], loc='upper right')
    plt.savefig(figfile)
