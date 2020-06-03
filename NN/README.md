# 神经网络入门实验

## 一 实验目的及要求

初步了解神经网络的入门知识：

+ 全连接
+ 激活函数
+ 独热编码
+ 损失函数
+ 梯度下降
+ 反向传播

## 二 实验环境

python == 3.6

Tensorflow == 2.2

## 三 实验内容

本着项目是最好的学习原则，本次实验使用最新的tensorflow2.0版本进行，鸢尾花的初级分类。搭建两层神经网络结构，实现前向传播，反向传播。并使用matplotlib可视化loss曲线。

### 实验步骤

1. sklearn输入iris数据集（150）
2. 分割训练集（120），测试集（30）
3. 因为数据量较少固不设置隐藏层结点。因为有四个特征固设置四个输入结点，三分类设置三个输出结点
4. 设置learning rate 为 0.1，epoch 为 500，batch 为 32
5. 使用with语句实现梯度下降，并进行后向传播
6. 记录loss值并计算正确率
7. matplotlib画出图像

## 四 实验结果与分析

最终

![image-20200603202035941](C:\Users\hi\AppData\Roaming\Typora\typora-user-images\image-20200603202035941.png)

![image-20200603202117120](C:\Users\hi\AppData\Roaming\Typora\typora-user-images\image-20200603202117120.png)

![image-20200603202127593](C:\Users\hi\AppData\Roaming\Typora\typora-user-images\image-20200603202127593.png)

## 五 源码

我将源码上传到[我的github中]()其中有详细注释