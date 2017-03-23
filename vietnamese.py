#coding:utf-8
from PIL import Image as image
from PIL import ImageFile
from tensorflow.python.framework import dtypes
from transform import resizeImg
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import os
import collections
import numpy as np

AlphaBeta_num = 29
AlphaBeta = ['.','A','Ă','Â','B','C','D','Đ','E','Ê','G','H','I','K','L','M','N','O','Ô','Ơ','P','Q','R','S','T','U','Ƣ','V','X','Y']

class DataSet(object):

  def __init__(self,
               images,
               labels,
               dtype=dtypes.float32,
               reshape=True):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)

    assert images.shape[0] == labels.shape[0], (
        'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
    self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
    if reshape:
      assert images.shape[3] == 1
      images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
    if dtype == dtypes.float32:
      # Convert from [0, 255] -> [0.0, 1.0].
      images = images.astype(np.float32)
      images = np.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""

    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

def to_binary(img_url):
    bn = np.arange(1200)
    bn = bn.reshape(40, 30, 1)
    img=image.open(img_url)
    width=img.size[0]
    height=img.size[1]
    cnt = 0
    for w in range(width):
        for h in range(height):
            r, g, b=img.getpixel((w, h))
            val = (r+g+b)/3
            #print(val)
            bn[w,h,0] = val
            
    #print(bn[20,20])
    return bn

def data(url, nums):
    fp = open(url,'r')
    line = fp.readline()
    outdata = np.arange(nums*1200)
    outdata = outdata.reshape(nums, 40, 30, 1)
    outlabel = np.zeros((nums, AlphaBeta_num))
    cnt = 0
    while line:
        ori_img = './Hnd/'+line
        ori_img = ori_img[0:34]
        dst_img = ori_img[:6] + 'Tny' + ori_img[9:31] + 'jpg'

        outdata[cnt] = to_binary(dst_img)
        outlabel[cnt,int(dst_img[16:19])-1] = 1
        line = fp.readline()
        print(cnt)
        if line == '':
            break
        cnt = cnt + 1
    fp.close()
    return (outdata,outlabel)

def test_data(url):
    outdata = np.arange(1200)
    outdata = outdata.reshape(1, 40, 30, 1)
    outlabel = np.zeros((1, AlphaBeta_num))
    outdata[0] = to_binary(url)
    outlabel[0,int(url[len(url)-11:len(url)-8])-1] = 1
    #outlabel[0,0] = 1
    return (outdata,outlabel)

def train():
    #训练数据
    data_train, label_train = data("./Hnd/trainyny.txt",1450)
    train = DataSet(data_train, label_train, dtype=dtypes.float32)
    data_test, label_test = data("./Hnd/testyny.txt",145)
    test = DataSet(data_test, label_test, dtype=dtypes.float32)
    Datasetsx = collections.namedtuple('Datasetsx', ['train', 'test'])
    Data = Datasetsx(train=train, test=test)
    #构造网络
    x = tf.placeholder(tf.float32, [None, 1200])
    y_actual = tf.placeholder(tf.float32, shape=[None, AlphaBeta_num])
    W = tf.Variable(tf.zeros([1200,AlphaBeta_num]))             #初始化权值W
    b = tf.Variable(tf.zeros([AlphaBeta_num]))                  #初始化偏置项b
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
    Target_Accuracy = 0.57
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
    data_tester, label_tester = test_data(testimg)
    tester = DataSet(data_tester, label_tester, dtype=dtypes.float32)
    #构造网络
    x = tf.placeholder(tf.float32, [None, 1200])
    y_actual = tf.placeholder(tf.float32, shape=[None, AlphaBeta_num])
    W = tf.Variable(tf.zeros([1200,AlphaBeta_num]))             #初始化权值W
    b = tf.Variable(tf.zeros([AlphaBeta_num]))                  #初始化偏置项b
    y_predict = tf.nn.softmax(tf.matmul(x,W) + b)               #加权变换并进行softmax回归，得到预测概率
    correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_actual,1))   #在测试阶段，测试准确度计算
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))                #多个批次的准确度均值
    saver = tf.train.Saver()
    #测试
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "./model/softmax.ckpt")
        accu = sess.run(y_actual, feed_dict={y_actual: tester.labels})
        actual = -1
        for i in range(0,AlphaBeta_num):
            if accu[0,i]>0:
                actual = i
                break
        predict = [-1,-1,-1]
        pr_mean = [-1,-1,-1]
        prdi = sess.run(y_predict, feed_dict={x: tester.images, y_actual: tester.labels})
        for i in range(0,AlphaBeta_num):
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
        print ("实际输入:", AlphaBeta[actual+1])
        print ("预测:", AlphaBeta[predict[0]+1],"，概率:", pr_mean[0])
        print ("预测:", AlphaBeta[predict[1]+1],"，概率:", pr_mean[1])
        print ("预测:", AlphaBeta[predict[2]+1],"，概率:", pr_mean[2])

train()
#try:
#    test('./Hnd/Tny/Sample007/img007-054.jpg')
#except Exception as e:
#    print(e)
