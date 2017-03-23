#coding:utf-8
from PIL import Image as image
from PIL import ImageFile
import os
import collections
import numpy as np


#等比例压缩图片
def resizeImg(**args):
    args_key = {'ori_img':'','dst_img':'','dst_w':'','dst_h':'','save_q':30}
    arg = {}
    for key in args_key:
        if key in args:
            arg[key] = args[key]
        
    im = image.open(arg['ori_img'])
    ori_w,ori_h = im.size
    widthRatio = heightRatio = None
    ratio = 1
    if (ori_w and ori_w > arg['dst_w']) or (ori_h and ori_h > arg['dst_h']):
        if arg['dst_w'] and ori_w > arg['dst_w']:
            widthRatio = float(arg['dst_w']) / ori_w #正确获取小数的方式
        if arg['dst_h'] and ori_h > arg['dst_h']:
            heightRatio = float(arg['dst_h']) / ori_h

        if widthRatio and heightRatio:
            if widthRatio < heightRatio:
                ratio = widthRatio
            else:
                ratio = heightRatio

        if widthRatio and not heightRatio:
            ratio = widthRatio
        if heightRatio and not widthRatio:
            ratio = heightRatio
            
        newWidth = int(ori_w * ratio)
        newHeight = int(ori_h * ratio)
    else:
        newWidth = ori_w
        newHeight = ori_h
        
    im.resize((newWidth,newHeight),image.ANTIALIAS).save(arg['dst_img'],quality=arg['save_q'])

def pre_handle():
    fp = open("E:/python/tensorflow/Hnd/all.txt",'r')
    line = fp.readline()
    while line:
        ori_img = './Hnd/'+line
        ori_img = ori_img[0:34]
        dst_img = ori_img[:6] + 'Tar' + ori_img[9:31] + 'jpg'
        dirs = dst_img[0:20]
        try:
            os.mkdir(dirs)
        except Exception as e:
            pass
        print(ori_img,dst_img,dirs)

        resizeImg(ori_img=ori_img,dst_img=dst_img,dst_w=40,dst_h=30,save_q=100)
        line = fp.readline()
        if line == '':
            break
    fp.close()

def trans(url):
    filelist = os.listdir(url)  
    filename = filelist[0]
    code = url[35:38]
    for i in range(1,56):
        xf = str(i)
        if (i < 10):
            xf = '0'+xf
        input_file = filename[:8]+xf+filename[10:]
        output_file = 'img'+code+'-0'+xf+filename[10:]
        #print(output_file)
        os.rename(url+input_file,url+output_file)

#trans('E:/python/tensorflow/Hnd/Yny/Sample011/')

# for i in range(1,30):
#     xi = str(i)
#     if (i < 10):
#         xi = '0'+xi
#     for j in range(51,56):
#         xj = str(j)
#         if (j < 10):
#             xj = '0'+xj
#         strs = 'Yny/Sample0'+xi+'/img0'+xi+'-0'+xj+'.png'
#         print(strs)

def pre_handle():
    fp = open("E:/python/tensorflow/Hnd/allyny.txt",'r')
    line = fp.readline()
    while line:
        ori_img = './Hnd/'+line
        ori_img = ori_img[0:34]
        dst_img = ori_img[:6] + 'Tny' + ori_img[9:31] + 'jpg'
        dirs = dst_img[0:20]
        try:
            os.mkdir(dirs)
        except Exception as e:
            pass
        print(ori_img,dst_img,dirs)

        resizeImg(ori_img=ori_img,dst_img=dst_img,dst_w=40,dst_h=30,save_q=100)
        line = fp.readline()
        if line == '':
            break
    fp.close()


# trans('E:/python/tensorflow/Hnd/Yny/Sample019/')
# pre_handle()
# resizeImg(ori_img='E:/python/tensorflow/Hnd/others/test.png',dst_img='E:/python/tensorflow/Hnd/others/testout.png',dst_w=40,dst_h=30,save_q=100)