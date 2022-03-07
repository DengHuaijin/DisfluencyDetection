## DisfluencyDetection 该项目是当前修士阶段的主要研究，对CSJ数据集里面语音的流畅或者非流畅部分进行判断以及检出

### 第一阶段是对一个10秒左右的data chunk整体进行分类，主要的特征分析，找出有效的特征

### SVM：

SVM模型发表在[ICASSP2020](https://ieeexplore.ieee.org/document/9053452)上，有兴趣的同学可以看一下，
其中有两个2分类任务，流畅 vs. 非流畅+中立 以及 非流畅 vs. 流畅+中立

### DNN:

#### 时序语音特征

用LSTM和时序语音特征的实验结果发表在[LREC2020](https://www.aclweb.org/anthology/2020.lrec-1.791/)上，
结合了文本特征和语音特征，结果用precision recall曲线进行了评估，和SVM一样的分类任务。

在这之后我们先后尝试了BiLSTM, Attention, CRNN等模型结构，最后发现BiLSTM+temporal mean pooling的结构表现最好，
详细可以参考[这篇论文](https://ieeexplore.ieee.org/document/8272614)。

#### 静态全局语音特征

静态全局特征就是用各种函数对时序特征沿着时间方向进行全局处理，比如max，min，mean..等等，
将LSTM直接换成DNN，其中jitter shimmer对非流畅语音分类效果提升显著，该结果投稿了SLT2021，目前还在审稿中。

### Deep Residual Network + BiLSTM

用语音的spectrogram特征和ResNet+BiLSTM结构进行端到端分类，参考论文：
[Detecting Multiple Speech Disfluencies using a Deep Residual Network with Bidirectional Long Short-Term Memory](
https://arxiv.org/abs/1910.12590)。和前面一样，数据还是10秒左右的音频文件，这里除了音频的spectrogram
没有使用前面的文本特征

网络的搭建和训练是基于NVIDIA的[OpenSeq2Seq](https://github.com/NVIDIA/OpenSeq2Seq)工具，
reidual block的具体实现在ResidualNetwork/models/resnet_block.py中

### 第二阶段是通过加入ASR，实时地根据识别结果输出单词级别的非流畅部分 (skip)

### 当前完成的工作是用深度端到端模型来替换前面的DNN和LSTM

下面所有模型均通过nvidia的[Nemo](https://github.com/NVIDIA/NeMo)来进行部署和训练
并且特征均采用mel bin为80的Mel Spectrogram

### SpeakerNet

[SpeakerNet](https://arxiv.org/pdf/2010.12653.pdf)是被提出来做说话人识别和声纹识别的模型，encoder部分是Quartznet模型，decoder部分直接用了stats pooling来进行x-vector的提取，
因此可以直接迁移到我们的分类任务中来

### Transformer，Conformer

[Conformer](https://arxiv.org/pdf/2005.08100.pdf?ref=https://githubhelp.com)是被提出来做ASR的模型, encoder部分是卷积和transformer的结合，原本decoder是可以直接是全连接+CTC，或者Attention decoder，在这里我们
将其替换为stats pooling+全连接；另外作为对比，直接将transformer作为encoder，去掉了卷积部分；2种模型结构如下图所示：

<div align="center">
<img src="https://github.com/DengHuaijin/DisfluencyDetection/tree/master/figs/model.jpg" width="600">
</div>
