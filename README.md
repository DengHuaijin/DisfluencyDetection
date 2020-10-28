# DisfluencyDetection 
## 该项目是当前修士阶段的主要研究，对CSJ数据集里面语音的流畅或者非流畅部分进行判断以及检出
## 第一阶段是对一个10秒左右的data chunk整体进行分类，主要的特征分析，找出有效的特征

### SVM：

SVM模型发表在[ICASSP2020](https://ieeexplore.ieee.org/document/9053452)上，有兴趣的同学可以看一下

其中有两个2分类任务，流畅 vs. 非流畅+中立 以及 非流畅 vs. 流畅+中立

### DNN:

#### 时序语音特征

用LSTM和时序语音特征的实验结果发表在[LREC2020](https://www.aclweb.org/anthology/2020.lrec-1.791/)上，
结合了文本特征和语音特征，结果用precision recall曲线进行了评估，和SVM一样的分类任务

在这之后我们先后尝试了BiLSTM, Attention, CRNN等模型结构，最后发现BiLSTM+temporal mean pooling的结构表现最好，
详细可以参考[这篇论文](https://ieeexplore.ieee.org/document/8272614)

#### 静态全局语音特征

静态全局特征就是用各种函数对时序特征沿着时间方向进行全局处理，比如max，min，mean..等等，
将LSTM直接换成DNN，其中jitter shimmer对非流畅语音分类效果提升显著，该结果投稿了SLT2021，目前还在审稿中。


## 第二阶段是通过加入ASR，实时地根据识别结果输出单词级别的非流畅部分
