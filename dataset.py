#!/usr/bin/python
#coding=utf-8

from PIL import Image as image
from PIL import ImageFile
import numpy as np
from tensorflow.python.framework import dtypes
from transform import resizeImg

# 数据集
class DataSet(object):

    AlphaBeta_num = 29
    AlphaBeta     = ['.','A','Ă','Â','B','C','D','Đ','E','Ê','G','H','I','K','L','M','N','O','Ô','Ơ','P','Q','R','S','T','U','Ƣ','V','X','Y']
    Width         = 40
    Height        = 30
    Area          = 1200
    train_file    = "./Data/Hnd/trainyny.txt"
    train_size    = 1450
    test_file     = "./Data/Hnd/testyny.txt"
    test_size     = 145

    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    def __init__(self,
               images,
               labels,
               dtype=dtypes.float32,
               reshape=True):
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)

        assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        if reshape:
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0],images.shape[1] * images.shape[2])
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

    # Return the next `batch_size` examples from this data set.
    def next_batch(self, batch_size):
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

    # 将图像转为(40x30)的图像
    def to_binary(img_url):
        bn = np.arange(DataSet.Area)
        bn = bn.reshape(DataSet.Width, DataSet.Height, 1)
        img=image.open(img_url)
        width=img.size[0]
        height=img.size[1]
        cnt = 0
        for w in range(width):
            for h in range(height):
                r, g, b=img.getpixel((w, h))
                val = (r+g+b)/3
                bn[w,h,0] = val
        return bn

    # 从Text中获取文件并转换
    def data_from_text(url, nums):
        fp = open(url,'r')
        line = fp.readline()
        outdata = np.arange(nums*DataSet.Area)
        outdata = outdata.reshape(nums, DataSet.Width, DataSet.Height, 1)
        outlabel = np.zeros((nums, DataSet.AlphaBeta_num))
        cnt = 0
        while line:
            ori_img = './Data/Hnd/'+line
            ori_img = ori_img[0:len(ori_img)-1]
            outdata[cnt] = DataSet_English.to_binary(ori_img)
            outlabel[cnt,int(ori_img[21:24])-1] = 1
            line = fp.readline()
            print(cnt)
            if line == '':
                break
            cnt = cnt + 1
        fp.close()
        return (outdata,outlabel)

    # 从List直接转换
    def data_from_array(img_data):
        outdata = np.arange(DataSet.Area)
        outdata = outdata.reshape(1, DataSet.Width, DataSet.Height, 1)
        outlabel = np.zeros((1, DataSet.AlphaBeta_num))
        bn = np.arange(DataSet.Area)
        bn = bn.reshape(DataSet.Width, DataSet.Height, 1)
        cnt = 0
        for w in range(DataSet.Width):
            for h in range(DataSet.Height):
                bn[w,h,0] = img_data[w+h*DataSet.Width]
                cnt = cnt+1
        outdata[0] = bn
        outlabel[0,0] = 1
        return (outdata,outlabel)

    # 获取图像转换
    def data_from_img(url):
        outdata = np.arange(DataSet.Area)
        outdata = outdata.reshape(1, DataSet.Width, DataSet.Height, 1)
        outlabel = np.zeros((1, DataSet.AlphaBeta_num))
        outdata[0] = DataSet.to_binary(url)
        outlabel[0,int(url[len(url)-11:len(url)-8])-1] = 1
        return (outdata,outlabel)

class DataSet_English(object):

    AlphaBeta_num = 62
    AlphaBeta     = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    Width         = 40
    Height        = 30
    Area          = 1200
    train_file    = "./Data/Eng/train.txt"
    train_size    = 3100
    test_file     = "./Data/Eng/test.txt"
    test_size     = 310

    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    def __init__(self,
               images,
               labels,
               dtype=dtypes.float32,
               reshape=True):
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)

        assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        if reshape:
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0],images.shape[1] * images.shape[2])
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

    # Return the next `batch_size` examples from this data set.
    def next_batch(self, batch_size):
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

    # 将图像转为(40x30)的图像
    def to_binary(img_url):
        bn = np.arange(DataSet_English.Area)
        bn = bn.reshape(DataSet_English.Width, DataSet_English.Height, 1)
        img=image.open(img_url)
        width=img.size[0]
        height=img.size[1]
        cnt = 0
        for w in range(width):
            for h in range(height):
                r, g, b=img.getpixel((w, h))
                val = (r+g+b)/3
                bn[w,h,0] = val
        return bn

    # 从Text中获取文件并转换
    def data_from_text(url, nums):
        fp = open(url,'r')
        line = fp.readline()
        outdata = np.arange(nums*DataSet_English.Area)
        outdata = outdata.reshape(nums, DataSet_English.Width, DataSet_English.Height, 1)
        outlabel = np.zeros((nums, DataSet_English.AlphaBeta_num))
        cnt = 0
        while line:
            ori_img = './Data/Eng/'+line
            ori_img = ori_img[0:len(ori_img)-1]
            outdata[cnt] = DataSet_English.to_binary(ori_img)
            outlabel[cnt,int(ori_img[21:24])-1] = 1
            line = fp.readline()
            print(cnt)
            if line == '':
                break
            cnt = cnt + 1
        fp.close()
        return (outdata,outlabel)

    # 从List直接转换
    def data_from_array(img_data):
        outdata = np.arange(DataSet_English.Area)
        outdata = outdata.reshape(1, DataSet_English.Width, DataSet_English.Height, 1)
        outlabel = np.zeros((1, DataSet_English.AlphaBeta_num))
        bn = np.arange(DataSet_English.Area)
        bn = bn.reshape(DataSet_English.Width, DataSet_English.Height, 1)
        cnt = 0
        for w in range(DataSet_English.Width):
            for h in range(DataSet_English.Height):
                bn[w,h,0] = img_data[w+h*DataSet_English.Width]
                cnt = cnt+1
        outdata[0] = bn
        outlabel[0,0] = 1
        return (outdata,outlabel)

    # 获取图像转换
    def data_from_img(url):
        outdata = np.arange(DataSet_English.Area)
        outdata = outdata.reshape(1, DataSet_English.Width, DataSet_English.Height, 1)
        outlabel = np.zeros((1, DataSet_English.AlphaBeta_num))
        outdata[0] = DataSet_English.to_binary(url)
        outlabel[0,int(url[len(url)-11:len(url)-8])-1] = 1
        return (outdata,outlabel)
