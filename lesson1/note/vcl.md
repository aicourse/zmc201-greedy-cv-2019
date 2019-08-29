# CV + Deep Learning

## 机器学习和人工神经网络的简介

* 机器学习 ～= 函数
  从输入到输出的黑盒函数，从输入到输出之间建立联系
  $ f(x, \theta^*) $

* 打比赛
  * 分类识别
  * 目标检测
  * 图像分割
  * 。。。
* 其他不同领域，如语言识别，下围棋。。。
* 特定数据集特定任务上超过普通人类到能力
  
## 深度学习的发展历程

* 1940 Electronic Brain
  仿生，人工设置权值
* 1950 Perception(感知器)
  自动通过数据学习权值，问题在于输出是线性分类器
* 80年代 Multiple-layered perception
  限制于当时到硬件和软件(数据), SVM
* 2010~
  算力的增加

### 感知器

* 对输入加权求和
* 激活函数是跃阶求导函数(step function)

### BP算法（误差向后传递） 1982

* 误差向后传递
* 可以梯度下降优化
* 可以自动训练多层网络

### 卷积+池化 1989

* 参数少于全联接层
* 更容易训练，节点比全联接少

### 2011年后

* 更好的初始化策略
* 新的ReLU
* 更多的训练数据

深度学习是一个综合技术，包含卷积、池化、全联接层，ReLU激活函数。

列举的几个典型例子：

* AlixNet
* VGGNet
* GoogLeNet
  大量使用Inception Module
  1x1卷积层的功能，可以把输入层的通道做缩放，从而提高效率
* ResNet
  跳跃层，Residual，上一层的输出和上两层的输入的差别，减少梯度消失，或者梯度爆炸

## 深度学习在计算机视觉中的应用

* 还好很多领域，比如压缩(卷积层的输出编程特征用于压缩)
* 生成图像描述
* 图像风格化转移
* 基于内容的图像

## 总结

## [准备开发环境](http://47.94.6.102/DeepLearningCV1/course-info/blob/master/README.md)

### 安装Anaconda Python, Tensorflow, Keras, PyTorch, OpenCV, nltk, Pillow

Python3.6, 对应的anaconda 下载

Mac

```shell
brew cask install anaconda
```

Linux

```shell
wget -O - https://repo.continuum.io/archive/Anaconda3-5.1.10-Linux-x86_64.sh | bash
```

### [Numpy学习](https://www.numpy.org.cn/article/basics/understanding_numpy.html)

### [Keras入门](https://github.com/yuan776/deep-learning-with-python-cn/blob/master/SUMMARY.md)

### 安装GPU驱动

### 安装[CUDA](https://developer.nvidia.com/cuda-downloads)

### 安装[cuDNN](https://developer.nvidia.com/cudnn)，测试Tensorflow，PyTorch

```python
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProtol(log_device_placement=True))
```
