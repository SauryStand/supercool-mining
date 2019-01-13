# -*- coding: utf-8 -*-

import cv2     # 用于获取视频
import math   # 用于数学计算
import matplotlib.pyplot as plt #%matplotlib inline
#%matplotlib inline
import pandas as pd
from keras.preprocessing import image   # 用于预处理图像
import numpy as np    # 用于数学计算
from keras.utils import np_utils
from skimage.transform import resize   # 用于调整图像大小



count = 0
videoFile = "data/Tom and jerry.mp4"
cap = cv2.VideoCapture(videoFile)   # 从给定路径中获取视频
frameRate = cap.get(5) #frame rate
x=1
while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % math.floor(frameRate) == 0):
        filename ="data/frame%d.jpg" % count;count+=1
        cv2.imwrite(filename, frame)
cap.release()
print ("Done!")