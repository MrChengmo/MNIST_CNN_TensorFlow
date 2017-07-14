# -*- coding:utf-8 -*-
import tensorflow as tf

#配置神经网络参数
INPUT_NODE = 784
OUTPUT_NODE =10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

#第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5
#第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV1_SIZE = 5
#全连接层的节点个数
FC_SIZE = 512

#定义卷积神经网络的前向传播过程；添加新参数train，用于区分训练过程和测试过程
#使用dropout方法，进一步提升模型可靠性和防止过拟合，只在训练时使用
def inference(input_tensor,train,regularizer):
    #声明第一层卷积层的变量并实现前向传播
    #通过使用不同的命名空间来隔离不同层的变量，不需担心重名问题
    #卷积层使用全0填充：输入28*28*1，输出28*28*32矩阵
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            "weight",[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],
            initializer = tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable(
            "biases",[CONV1_DEEP],initializer = tf.constant_initializer(0.0))

        #使用边长为5，深度为32的过滤器，步长1，全0填充
        conv1 = tf.nn.conv2d(
            input_tensor,conv1_weights,strides=[1,1,1,1],padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))

    #实现第二层池化层的前向传播过程
    #使用最大池化层，边长为2，全0填充，步长为2
    #输入28*28*32，输出14*14*32
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(
            relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')