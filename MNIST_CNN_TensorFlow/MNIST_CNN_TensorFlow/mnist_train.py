# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import numpy as np

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
#模型保存路径和文件名
MODEL_SAVE_PATH = "E:/path/to/model/"
MODEL_NAME = "model.ckpt"

#定义神经网络训练过程
def train(mnist):
    #定义输入输出place holder
    #调整输出数据格式，输入为一个四维矩阵
    x = tf.placeholder(
        tf.float32,
        [BATCH_SIZE,
         mnist_inference.IMAGE_SIZE,
         mnist_inference.IMAGE_SIZE,
         mnist_inference.NUM_CHANNELS],name = 'x-input')
    y_= tf.placeholder(
        tf.float32,[None,mnist_inference.OUTPUT_NODE],name = 'y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    #直接使用mnist_interence中定义的前向传播过程
    y = mnist_inference.inference(x,regularizer)
    global_step = tf.Variable(0,trainable = False)

    #定义损失函数、学习率、滑动平均操作以及训练过程
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op = variable_averages.apply(
        tf.trainable_variables())
    #定义交叉熵及其平均
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    #定义损失函数
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    #定义学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples/BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase = True)
    #定义训练轮数
    training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss,global_step = global_step)
    with tf.control_dependencies([training_step,variable_averages_op]):
        train_op = tf.no_op(name = 'train')

    #初始化tensorflow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        #在训练过程中不在测试模型在验证数据上的表现，验证和测试的过程将会有一个独立的程序完成
        for i in range(TRAINING_STEPS):
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            #将训练数据调整为一个四维矩阵
            reshaped_xs=np.reshape(xs,(
                BATCH_SIZE,mnist_inference.IMAGE_SIZE,mnist_inference.IMAGE_SIZE,mnist_inference.NUM_CHANNELS))
            _,loss_vaule,step = sess.run([train_op,loss,global_step],feed_dict={x:reshaped_xs,y_:ys})
            #每1000轮保存一次模型
            if i%1000 == 0:
                #输出当前的训练情况，这里输出在当前训练batch上的损失函数大小
                #通过损失函数可以大概了解训练情况
                #在验证集上的正确率信息会单独程序完成。
                print("After %d training step,loss on training "
                      "batch is %g"%(step,loss_vaule))
                #保存当前的模型，这里加上了global_step的参数，可以让每个被保存模型的文件名末尾加上训练的轮数
                saver.save(
                    sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)
#定义主函数
def main(argv = None):
    mnist= input_data.read_data_sets("/tmp/data",one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()