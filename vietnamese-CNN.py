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
import math

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





# 产生随机变量，符合 normal 分布
# 传递 shape 就可以返回weight和bias的变量
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)    
    return tf.Variable(initial)                            

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 定义2维的 convolutional 图层
def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    # strides 就是跨多大步抽取信息
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')        

# 定义 pooling 图层
def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    # 用pooling对付跨步大丢失信息问题
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    
# 定义 pooling 图层
def max_pool_5x5(x):
    # stride [1, x_movement, y_movement, 1]
    # 用pooling对付跨步大丢失信息问题
    return tf.nn.max_pool(x, ksize=[1,5,5,1], strides=[1,5,5,1], padding='SAME')    

def train():
    #训练数据
    data_train, label_train = data("./Hnd/trainyny.txt",1450)
    train = DataSet(data_train, label_train, dtype=dtypes.float32)
    data_test, label_test = data("./Hnd/testyny.txt",145)
    test = DataSet(data_test, label_test, dtype=dtypes.float32)
    Datasetsx = collections.namedtuple('Datasetsx', ['train', 'test'])
    Data = Datasetsx(train=train, test=test)

    #输入
    xs = tf.placeholder(tf.float32, [None, 1200])
    ys = tf.placeholder(tf.float32, shape=[None, AlphaBeta_num])
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(xs, [-1, 40, 30, 1])

    ## 1. conv1 layer ##
    #  把x_image的厚度1加厚变成了32
    W_conv1 = weight_variable([5, 5, 1, 32])                 # patch 5x5, in size 1, out size 32
    b_conv1 = bias_variable([32])
    # 构建第一个convolutional层，外面再加一个非线性化的处理relu
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)             # output size 40x30x32
    # 经过pooling后，长宽缩小为14x14
    h_pool1 = max_pool_2x2(h_conv1)                                     # output size 20x15x32

    ## 2. conv2 layer ##
    # 把厚度32加厚变成了64
    W_conv2 = weight_variable([5, 5, 32, 64])                 # patch 5x5, in size 32, out size 64
    b_conv2 = bias_variable([64])
    # 构建第二个convolutional层
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)             # output size 14x14x64
    # 经过pooling后，长宽缩小为7x7
    h_pool2 = max_pool_5x5(h_conv2)                                     # output size 4x3x64

    ## 3. func1 layer ##
    # 飞的更高变成1024
    W_fc1 = weight_variable([4*3*64, 1024])
    b_fc1 = bias_variable([1024])
    # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
    # 把pooling后的结果变平
    h_pool2_flat = tf.reshape(h_pool2, [-1, 4*3*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    ## 4. func2 layer ##
    # 最后一层，输入1024，输出size 10，用 softmax 计算概率进行分类的处理
    W_fc2 = weight_variable([1024, AlphaBeta_num])
    b_fc2 = bias_variable([AlphaBeta_num])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


    # the error between prediction and real data
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                                  reduction_indices=[1]))       # loss
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # init
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    Target_Accuracy = 0.9
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(50000):                                  #训练阶段，迭代50000次
            batch_X, batch_Y =Data.train.next_batch(100)
            sess.run(train_step, feed_dict={xs: batch_X, ys: batch_Y, keep_prob: 0.5})
            accu = 0
            if(i%100==0):                                        #每训练100次，测试一次
                v_xs = Data.test.images
                v_ys = Data.test.labels
                y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
                correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                accu = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
                print ("accuracy:", accu)
            if(accu>Target_Accuracy):
                break
        saver.save(sess, "./save.ckpt")

def test(testimg):
    #输入图像
    data_tester, label_tester = test_data(testimg)
    tester = DataSet(data_tester, label_tester, dtype=dtypes.float32)

    #输入
    xs = tf.placeholder(tf.float32, [None, 1200])
    ys = tf.placeholder(tf.float32, shape=[None, AlphaBeta_num])
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(xs, [-1, 40, 30, 1])

    ## 1. conv1 layer ##
    #  把x_image的厚度1加厚变成了32
    W_conv1 = weight_variable([5, 5, 1, 32])                 # patch 5x5, in size 1, out size 32
    b_conv1 = bias_variable([32])
    # 构建第一个convolutional层，外面再加一个非线性化的处理relu
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)             # output size 40x30x32
    # 经过pooling后，长宽缩小为14x14
    h_pool1 = max_pool_2x2(h_conv1)                                     # output size 20x15x32

    ## 2. conv2 layer ##
    # 把厚度32加厚变成了64
    W_conv2 = weight_variable([5, 5, 32, 64])                 # patch 5x5, in size 32, out size 64
    b_conv2 = bias_variable([64])
    # 构建第二个convolutional层
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)             # output size 14x14x64
    # 经过pooling后，长宽缩小为7x7
    h_pool2 = max_pool_5x5(h_conv2)                                     # output size 4x3x64

    ## 3. func1 layer ##
    # 飞的更高变成1024
    W_fc1 = weight_variable([4*3*64, 1024])
    b_fc1 = bias_variable([1024])
    # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
    # 把pooling后的结果变平
    h_pool2_flat = tf.reshape(h_pool2, [-1, 4*3*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    ## 4. func2 layer ##
    # 最后一层，输入1024，输出size 10，用 softmax 计算概率进行分类的处理
    W_fc2 = weight_variable([1024, AlphaBeta_num])
    b_fc2 = bias_variable([AlphaBeta_num])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


    # the error between prediction and real data
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                                  reduction_indices=[1]))       # loss
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # init
    sess = tf.Session()
    saver = tf.train.Saver()

    #测试
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "./save.ckpt")
        accu = sess.run(ys, feed_dict={ys: tester.labels})
        actual = -1
        for i in range(0,AlphaBeta_num):
            if accu[0,i]>0:
                actual = i
                break
        predict = [-1,-1,-1]
        pr_mean = [-1,-1,-1]
        v_xs = tester.images
        v_ys = tester.labels
        prdi = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
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
        '''
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
        '''
#train()
try:
    test('./Hnd/Tny/Sample007/img007-054.jpg')
except Exception as e:
    print(e)
