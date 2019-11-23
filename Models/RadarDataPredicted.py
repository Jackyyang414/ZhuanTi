# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 13:54:41 2019

@author: jacky
"""
from PIL import Image 
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import numpy as np
import time
RadarImagePath= "D:/ZhuanTi/RadarEchoDatabase/RadarImage/"#雷達迴波路徑
Weight=100#降低畫素
Height=100#降低畫素
OsList=os.listdir(RadarImagePath)
OsList.reverse()
SEQUENCE=np.array([])
BASIC_SEQUENCE=np.array([])
NEXT_SEQUENCE=np.array([])
NUMBER=0
def ImageInitialize(RadarImage):
    Picture=Image.open(RadarImage)
    Picture=Picture.resize((Weight,Height),Image.ANTIALIAS)#ANTIALIAS=平滑濾波
    #Picture=Picture.convert('L')#轉黑白
    Picture.save('D:/ZhuanTi/RadarEchoDatabase/Temp/Temp.png')  # 非保留
    Data=np.array(Picture.getdata()).reshape(Weight,Height,1)
    return Data
for RadarFile in OsList[0:241]:
    ImageArray=ImageInitialize(os.path.join(RadarImagePath,RadarFile))
    SEQUENCE=np.append(SEQUENCE,ImageArray)
    NUMBER+=1
    print(NUMBER)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
SEQUENCE=SEQUENCE.reshape(NUMBER,Weight*Height)
for i in SEQUENCE:
    for j in range(int(len(i))):
        if i[j] < 50:
            i[j] = 0
np.savez('D:/ZhuanTi/Models/SequenceArray.npz',SequenceArray=SEQUENCE)
print('Data saved.')
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

    