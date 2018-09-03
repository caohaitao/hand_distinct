__author__ = 'ck_ch'
import os
import cv2
import numpy as np

def read_one_data(file_path,width,height):
    im = cv2.imread(file_path)
    data = (cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)/255.0).reshape(1,width,height).astype(np.float32)
    return data
    # cv2.imshow("image",gray)
    # cv2.waitKey(0)

def read_datas(width,height):
    dir = 'data\\'
    for (root,dirs,files) in os.walk(dir):
        print(len(files))
        res = np.ndarray(shape=(len(files),1,width,height),dtype='float32')
        label = np.ndarray(shape=(len(files)),dtype='int64')
        i = 0
        for item in files:
            file_path = format("%s%s"%(dir,item))
            res[i] = read_one_data(file_path,width,height)
            sl = item.replace('.jpg','')
            sls = sl.split('_')
            label[i] = int(sls[1])
            i = i+1
    return res,label
