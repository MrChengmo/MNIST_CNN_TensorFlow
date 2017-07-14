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
#ģ�ͱ���·�����ļ���
MODEL_SAVE_PATH = "E:/path/to/model/"
MODEL_NAME = "model.ckpt"

#����������ѵ������
def train(mnist):
    #�����������place holder
    #����������ݸ�ʽ������Ϊһ����ά����
    x = tf.placeholder(
        tf.float32,
        [BATCH_SIZE,
         mnist_inference.IMAGE_SIZE,
         mnist_inference.IMAGE_SIZE,
         mnist_inference.NUM_CHANNELS],name = 'x-input')
    y_= tf.placeholder(
        tf.float32,[None,mnist_inference.OUTPUT_NODE],name = 'y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    #ֱ��ʹ��mnist_interence�ж����ǰ�򴫲�����
    y = mnist_inference.inference(x,regularizer)
    global_step = tf.Variable(0,trainable = False)

    #������ʧ������ѧϰ�ʡ�����ƽ�������Լ�ѵ������
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op = variable_averages.apply(
        tf.trainable_variables())
    #���彻���ؼ���ƽ��
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    #������ʧ����
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    #����ѧϰ��
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples/BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase = True)
    #����ѵ������
    training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss,global_step = global_step)
    with tf.control_dependencies([training_step,variable_averages_op]):
        train_op = tf.no_op(name = 'train')

    #��ʼ��tensorflow�־û���
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        #��ѵ�������в��ڲ���ģ������֤�����ϵı��֣���֤�Ͳ��ԵĹ��̽�����һ�������ĳ������
        for i in range(TRAINING_STEPS):
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            #��ѵ�����ݵ���Ϊһ����ά����
            reshaped_xs=np.reshape(xs,(
                BATCH_SIZE,mnist_inference.IMAGE_SIZE,mnist_inference.IMAGE_SIZE,mnist_inference.NUM_CHANNELS))
            _,loss_vaule,step = sess.run([train_op,loss,global_step],feed_dict={x:reshaped_xs,y_:ys})
            #ÿ1000�ֱ���һ��ģ��
            if i%1000 == 0:
                #�����ǰ��ѵ���������������ڵ�ǰѵ��batch�ϵ���ʧ������С
                #ͨ����ʧ�������Դ���˽�ѵ�����
                #����֤���ϵ���ȷ����Ϣ�ᵥ��������ɡ�
                print("After %d training step,loss on training "
                      "batch is %g"%(step,loss_vaule))
                #���浱ǰ��ģ�ͣ����������global_step�Ĳ�����������ÿ��������ģ�͵��ļ���ĩβ����ѵ��������
                saver.save(
                    sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)
#����������
def main(argv = None):
    mnist= input_data.read_data_sets("/tmp/data",one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()