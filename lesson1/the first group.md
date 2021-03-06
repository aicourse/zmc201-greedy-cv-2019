# CV第一章学习笔记

## 机器学习、深度学习简介

*机器学习：通过算法使得机器能从大量历史数据中学习规律从而对新的样本做预测。如语音识别，构建一个函数，输入一段语音信号，输出一段文字；如图片识别，构
建一个函数，输入一张图片，输出该图片的识别结果
*深度学习是机器学习的子集，是基于神经网络的一种机器学习算法，主要有以下优势：
1) 深度神经网络在多个计算机视觉任务的比赛中获得了很好的成绩；
2) 深度神经网络模型可以用于不同的领域 (语言识别, 下围棋, 玩游戏, 预测比赛结果,计算机视觉等等)；
3) 在特定数据集特定任务上面超过普通人类的能力；
4) 活跃的社区, 大量的资源；
5) 计算机硬件软件技术的进步提供了技术能力支撑。

## 几种典型的深度神经网络结构
1) AlexNet（2012），共有 62378344 个权值 (250MB)，24 层。
2) VGGNet (2013)，102908520 个权值 (412MB)，23 层。
3) GoogLeNet(2014),6998552 个权值 (28MB)，143 层，首次提出了 Inception 的概念，主要贡献有两个：一是使用 1x1 的卷积来进行升降维；二是在多个尺寸上同时进行卷积再聚合，即提取了更多的信息。
4）ResNet (2015)，创造了 Residual 残差学习,152 层

## 深度学习在计算机视觉中的应用
* 深度学习在计算机视觉中主要有四个方面：图片分类（单目标），目标定位（单目标），目标检测（多目标），语义分割（多目标）。
*除此之外，还有压缩，Auto-encoders,，Self-organizing maps，生成图像描述 Image Captioning，结合递归神经网络 NLP，图像风格化转移 Image Stylization，基于内容的图像 Image Retrieval 等方面的应用
1) 分类识别：每一副图包含一种类别的物体，最后一层是 Softmax 层
2) 目标定位： Bounding box Regression 回归 + Class Recognition 分类
3) 目标检测：一幅图包含多个物体, 物体属于多种类别，常用算法有 R-CNN、Faster R-CNN、Faster R-CNN、YOLO 、SSD 等
4) 语义分割：计算机根据图像的语义来进行分割
总结，相较于机器学习，深度学习更强大, 自动从数据学习特征, 无需手工提取特征；节约时间, 而且更好 (自动提取了空间和图像结合的特征, 人脑无法想象)；更深的网络具有较好的能力。同时深度学习的问题如下：需要防止过拟合，需要更大的数据量，对数据质量要求较高，需要进行数据标注。

## 深度学习训练环境的配置

## NVIDIA GPU + 对应驱动程序 + CUDA + cuDNN + TensorFlow（PyTorch）。
首先，Google 或者百度一下你的电脑是否配有 N 卡，如果没有（比如我），那就不需要考虑下面的环境配置过程了。但是还有别的出路，比如上 Google Colab（需科学上网）、Kaggle 上自己找显卡资源吧。接下来的教程适用于电脑配备有 N 卡的用户

### 1) 下载驱动：进入 Nivdia 官网找到适合你电脑显卡的驱动并下载。
[地址]（https://www.geforce.com/drivers）

### 2）下载安装 CUDA：
这是一种由 NVIDIA 推出的通用并行计算架构，该架构使 GPU 能够解决复杂的计算问题，加速矩阵计算，并且 CUDA 是向后兼容的，即旧版本用新
的系统打开依然可以用。只有少数 GPU 有不适用的 CUDA 版本。
[查询 GPU 适用于哪一种 CUDA 的链接:](https://developer.nvidia.com/cuda-gpus)
[CUDA 下载链接（建议使用本地安装包，稳定）:](https://developer.nvidia.com/cuda-downloads)

 ### 3)下载安装 cuDNN：
 cuDNN 是专门用于神经网络加速计算的工具，与 CUDA 配套.
 [cudnn下载地址](https://developer.nvidia.com/cudnn)
 
 ### 4) 安装 TensorFlow GPU 版本或者 PyTorch GPU 版本：从官网找到对应的 pip 安装代码后，通过命令行安装。

#### 检测 TesorFlow GPU 是否有效代码
```javascript
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
```

#### 检测 PyTorch GPU 是否有效代码
```javascript
import torch as T
T.cuda.current_device()
T.cuda.device_count()
T.cuda.get_device_name(0)
```

