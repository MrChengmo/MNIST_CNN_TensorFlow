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
CONV2_SIZE = 5
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
            "bias",[CONV1_DEEP],initializer = tf.constant_initializer(0.0))

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

    #声明第三层卷积层的前向传播过程
    #输入为14*14*32，输出为14*14*64
    with tf.name_scope('layer3-conv2'):
        conv2_weights = tf.get_variable(
            "weight",[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],
            initializer = tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable(
            "bias",[CONV2_DEEP],
            initializer = tf.constant_initializer(0.0))

        #使用边长为5，深度为64的过滤器，过滤器步长为1，全0填充
        conv2 = tf.nn.conv2d(
            pool1,conv2_weights,strides=[1,1,1,1],padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))

    #声明第四层池化层的前向传播过程
    #输入为14*14*64，输出为7*7*64
    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(
            relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

        #将第四层池化层的输出转化为第五层全连接层的输入格式，需要将7*7*64拉直为一个向量
        #pool2.get_shape可以得到第四层输出矩阵的维度而不需要手工计算。
        #因为每一层神经网络的输入输出都为一个BATCH矩阵，所以这里得到的维度也包含了一个batch中数据的个数
        pool_shape = pool2.get_shape().as_list()

        #计算将矩阵拉直成向量之后的长度，这个长度就是矩阵长宽深的乘积，
        #pool_shape[0]为一个batch中数据的个数
        nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]

        #通过tf.reshape将第四层的输出变为向量
        reshaped = tf.reshape(pool2,[pool_shape[0],nodes])

    #声明第五层全连接层的变量并实现前向传播
    #输入向量长度3136，输出向量长度512
    #此处加入dropout层
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable(
            "weight",[nodes,FC_SIZE],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        #全连接层的权重加入正则化
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1_biases = tf.get_variable(
                "bias",[FC_SIZE],initializer=tf.constant_initializer(0.1))

        fc1= tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc1_biases)
        if train: fc1=tf.nn.dropout(fc1,0.5)

    #声明第六层全连接层的变量并实现前向传播
    #输入为长度512的向量，输出为10长度的向量
    #通过softmax后得到最终的分类结果
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable(
            "weight",[FC_SIZE,NUM_LABELS],
            initializer = tf.truncated_normal_initializer(stddev=0.1))
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(fc2_weights))
        fc2_biases = tf.get_variable(
            "bias",[NUM_LABELS],
            initializer= tf.constant_initializer(0.1))
        logit = tf.matmul(fc1,fc2_weights)+fc2_biases

    #返回第六层的输出
    return logit