#coding:utf-8

import tensorflow as tf
import os
import collections
import numpy as np
from tensorflow.python.framework import dtypes
import math
from dataset import DataSet

#构造网络
Target_Accuracy = 0.57

x = tf.placeholder(tf.float32, [None, 1200])
y_actual = tf.placeholder(tf.float32, shape=[None, DataSet.AlphaBeta_num])
W = tf.Variable(tf.zeros([1200,DataSet.AlphaBeta_num]))             #初始化权值W
b = tf.Variable(tf.zeros([DataSet.AlphaBeta_num]))                  #初始化偏置项b
y_predict = tf.nn.softmax(tf.matmul(x,W) + b)               #加权变换并进行softmax回归，得到预测概率
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_actual*tf.log(y_predict),reduction_indices=1))   
                                                            #求交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)   
                                                            #用梯度下降法使得残差最小
correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_actual,1))   
                                                            #在测试阶段，测试准确度计算
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))                
                                                            #多个批次的准确度均值
saver = tf.train.Saver()

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
        sess.run(tf.global_variables_initializer())
        for i in range(50000):                                  #训练阶段，迭代50000次
            batch_xs, batch_ys = Data.train.next_batch(50)      #按批次训练，每批50行数据
            sess.run(train_step, feed_dict={x: batch_xs, y_actual: batch_ys})   
                                                                #执行训练
            accu = 0
            if(i%50==0):                                        #每训练100次，测试一次
                accu = sess.run(accuracy, feed_dict={ x: Data.test.images, 
                                                      y_actual: Data.test.labels})
                print ("accuracy:", accu)
            if(accu>Target_Accuracy):
                break
        saver.save(sess, "./model/softmax.ckpt")

def test(testimg):
    #输入图像
    data_tester, label_tester = DataSet.data_from_img(testimg)
    tester = DataSet(data_tester, label_tester, dtype=dtypes.float32)

    #测试
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "./model/softmax.ckpt")
        accu = sess.run(y_actual, feed_dict={y_actual: tester.labels})
        actual = -1
        for i in range(0,DataSet.AlphaBeta_num):
            if accu[0,i]>0:
                actual = i
                break
        predict = [-1,-1,-1]
        pr_mean = [-1,-1,-1]
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
        print ("实际输入:", DataSet.AlphaBeta[actual+1])
        print ("预测:", DataSet.AlphaBeta[predict[0]+1],"，概率:", pr_mean[0])
        print ("预测:", DataSet.AlphaBeta[predict[1]+1],"，概率:", pr_mean[1])
        print ("预测:", DataSet.AlphaBeta[predict[2]+1],"，概率:", pr_mean[2])

def predic(input_img):
    #输入图像
    data_tester, label_tester = DataSet.data_from_array(input_img)
    tester = DataSet(data_tester, label_tester, dtype=dtypes.float32)

    #测试
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "./model/softmax.ckpt")
        accu = sess.run(y_actual, feed_dict={y_actual: tester.labels})
        actual = -1
        for i in range(0,DataSet.AlphaBeta_num):
            if accu[0,i]>0:
                actual = i
                break
        predict = [-1,-1,-1]
        pr_mean = [-1,-1,-1]
        pr_rtn = ['a','b','c']
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


#train()
#try:
#    test('./Hnd/Tny/Sample007/img007-054.jpg')
#except Exception as e:
#    print(e)

#pr_mean,pr_rtn = predic([255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,0,118,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,162,0,0,0,0,0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,44,0,255,255,255,0,0,0,0,0,0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,255,255,255,255,255,255,255,51,0,0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,255,255,255,255,255,255,255,255,255,0,32,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,255,255,255,255,255,255,255,255,255,0,8,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,255,255,255,255,255,255,255,255,255,0,17,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,255,255,255,255,255,255,255,255,0,0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,255,255,255,255,255,255,255,109,0,101,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,239,255,255,255,255,255,255,255,0,0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,255,255,255,255,255,255,135,0,0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,0,0,15,15,15,9,0,0,0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,255,0,0,0,99,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,255,255,255,0,0,0,0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,255,255,255,255,255,7,0,0,0,110,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,84,255,255,255,255,255,255,255,255,0,0,0,0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,53,255,255,255,255,255,255,255,255,255,255,0,0,0,138,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,52,255,255,255,255,255,255,255,255,255,255,255,250,0,0,0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,106,255,255,255,255,255,255,255,255,255,255,255,255,255,251,203,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255])
#print(pr_rtn)