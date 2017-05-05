#!/usr/bin/python
#coding=utf-8

import math
import sys
import json
import os
import collections
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from dataset import DataSet

#构造网络
Target_Accuracy = 0.57
global cnt
global sess
cnt = 0

with tf.variable_scope("Softmax"):
    x = tf.placeholder(tf.float32, [None, 1200])
    tf.summary.histogram('input_x', x)
    y_actual = tf.placeholder(tf.float32, shape=[None, DataSet.AlphaBeta_num])
    tf.summary.histogram('input_y', y_actual)
    W = tf.Variable(tf.zeros([1200,DataSet.AlphaBeta_num]))             #初始化权值W
    tf.summary.histogram('weight', W)
    b = tf.Variable(tf.zeros([DataSet.AlphaBeta_num]))                  #初始化偏置项b
    tf.summary.histogram('bias', b)
    y_predict = tf.nn.softmax(tf.matmul(x,W) + b)               #加权变换并进行softmax回归，得到预测概率
    tf.summary.histogram('predict_y', y_predict)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_actual*tf.log(y_predict),reduction_indices=1))   
                                                                #求交叉熵
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)   
                                                                #用梯度下降法使得残差最小
    correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_actual,1))   
                                                                #在测试阶段，测试准确度计算
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))                
                                                                #多个批次的准确度均值
    #tf.summary.histogram('cross', correct_prediction)
    
    
    

def train():
    #训练数据
    data_train, label_train = DataSet.data_from_text("./Hnd/trainyny.txt",1450)
    train = DataSet(data_train, label_train, dtype=dtypes.float32)
    data_test, label_test = DataSet.data_from_text("./Hnd/testyny.txt",145)
    test = DataSet(data_test, label_test, dtype=dtypes.float32)
    Datasetsx = collections.namedtuple('Datasetsx', ['train', 'test'])
    Data = Datasetsx(train=train, test=test)

    #训练过程
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.getcwd() + '/log',sess.graph)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver({'s_w': W, 's_b': b})
        for i in range(500000):                                  #训练阶段，迭代50000次
            batch_xs, batch_ys = Data.train.next_batch(50)      #按批次训练，每批50行数据
            sess.run(train_step, feed_dict={x: batch_xs, y_actual: batch_ys})   
                                                                #执行训练
            accu = 0
            if(i%50==0):                                        #每训练100次，测试一次
                accu = sess.run(accuracy, feed_dict={ x: Data.test.images, 
                                                      y_actual: Data.test.labels})
                print ("accuracy:", accu)
                summary = sess.run(merged, feed_dict={ x: Data.test.images, 
                                                      y_actual: Data.test.labels})
                train_writer.add_summary(summary, i)
            if(accu>Target_Accuracy):
                break
        saver.save(sess, "./model/softmax.ckpt")

def predic(input_img):
    #输入图像
    data_tester, label_tester = DataSet.data_from_array(input_img)
    tester = DataSet(data_tester, label_tester, dtype=dtypes.float32)

    predict = [-1,-1,-1]
    pr_mean = [-1,-1,-1]
    pr_rtn = ['a','b','c']
    #测试
    with tf.variable_scope("Softmax"):
        global cnt
        global sess
        if cnt == 0:
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver({'s_w': W, 's_b': b})
            saver.restore(sess, "./model/softmax.ckpt")
            cnt = 1
            print("Freshed")

        accu = sess.run(y_actual, feed_dict={y_actual: tester.labels})
        actual = -1
        for i in range(0,DataSet.AlphaBeta_num):
            if accu[0,i]>0:
                actual = i
                break
        prdi = sess.run(y_predict, feed_dict={x: tester.images, y_actual: tester.labels})
        for i in range(0,DataSet.AlphaBeta_num):
            if prdi[0,i]> pr_mean[2]:
                predict[2] = i
                pr_mean[2] = prdi[0,i]
            if prdi[0,i]> pr_mean[1]:
                predict[2] = predict[1]
                pr_mean[2] = pr_mean[1]
                predict[1] = i
                pr_mean[1] = prdi[0,i]
            if prdi[0,i]> pr_mean[0]:
                predict[1] = predict[0]
                pr_mean[1] = pr_mean[0]
                predict[0] = i
                pr_mean[0] = prdi[0,i]
        ald = pr_mean[0] + pr_mean[1] + pr_mean[2]
        pr_mean[0] = pr_mean[0] / ald
        pr_mean[1] = pr_mean[1] / ald
        pr_mean[2] = pr_mean[2] / ald
        pr_rtn[0]  = DataSet.AlphaBeta[predict[0]+1]
        pr_rtn[1]  = DataSet.AlphaBeta[predict[1]+1]
        pr_rtn[2]  = DataSet.AlphaBeta[predict[2]+1]

    return pr_mean,pr_rtn

if __name__ == '__main__':
    train()