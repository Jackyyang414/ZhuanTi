# -*- coding: utf-8 -*-
"""
Created on Sun May  5 02:03:01 2019

@author: jacky
"""

import numpy as np
#from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.layers import Conv3D,MaxPooling3D,ConvLSTM2D
from keras.layers.normalization import BatchNormalization
#from keras.activations import relu
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import os
import matplotlib.pyplot as plt
import time
from keras import backend as K
K.set_image_dim_ordering('th')
EpochSize=100
SEQUENCE=3
BATCH_INDEX=0
#EpochSize=5000
TimeStart=time.time()
np.random.seed(10)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#選定跑模型的GPU
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'#顯示所有訊息
gpu_options=tf.GPUOptions(allow_growth=True)#允許自動增加GPU的使用空間
sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
KTF.set_session(sess)
RainPath=["E:/2017_gauge/JUN",
          "E:/2017_gauge/JUL",
          "E:/2017_gauge/AUG"]#六七八月雨量路徑
RadarPath=["D:/ZhuanTi/RadarEchoDatabase/PreprocessedRadarValue/Jun",
           "D:/ZhuanTi/RadarEchoDatabase/PreprocessedRadarValue/Jul",
           "D:/ZhuanTi/RadarEchoDatabase/PreprocessedRadarValue/Aug"]#六七八月雷達路徑
Month=["June","July","August"]
DateFormat="%Y%m%d%H%M"
Keyword="C0F9N0"#大里雨量站代碼
Quantity=4320+4462+4464-27-3#(6月+7月+8月-missing data)
LabelQuantity=Quantity+3
RList=np.zeros(([Quantity,21,21])) 
RainList=[]
RainTimeList=[]
RadarTimeList=[]
a=0
print("Model Start")
print("Loading RadarValue")
for index in range(len(RadarPath)):#載入雷達迴波值
    PRadarFiles=os.listdir(RadarPath[index])
    for PRadarValue in PRadarFiles:
        F=open(RadarPath[index]+'/'+PRadarValue,'rb')
        RadarContent=np.load(F)#RadarContent's shape=(21,21,21)
        RadarArray=RadarContent.reshape([21,21,21]).astype(np.float64)
        RadarArray2=RadarArray[0:1,:,:]
        RList[a]=RadarArray2
        a+=1
print("Finish loading RadarValue")
print("Labeling")
#貼Label
Label=np.zeros((LabelQuantity,3))
NewLabel=np.zeros((Quantity,3))
for i in range(len(RainPath)):
    RainList=os.listdir(RainPath[i])
    for RainFile in RainList:
        RainValue=open(RainPath[i]+"/"+RainFile)#載入雨量資料
        content=RainValue.readlines()[:-2]
        for line in content:
            line=line.replace('-999.00','0.00')
            line=line.replace('-998.00','0.00')
            if(line[0:6]=='C0F9N0'):
              line.split(" ")
              rain=float(line[49:53])
              if 0.34>rain>=0.0:#Label 0=無雨 1=小雨 2=大雨
                    RainLabel=[1,0,0]
                    np.append(Label,RainLabel)
              if 6.6>=rain>=0.34:
                    RainLabel=[0,1,0]
                    np.append(Label,RainLabel)
              if rain>6.6:
                    RainLabel=[0,0,1]
                    np.append(Label,RainLabel)
print("Finish Labeling")
DeleteLabelIndex=[0,1,2]
NewLabel=np.delete(Label,DeleteLabelIndex,axis=0)
#np.save("D:/ZhuanTi/RainDataBase/RainLabel.npy",NewLabel)
#np.savez("D:/ZhuanTi/RadarEchoDatabase/PreprocessedRadarValue.npy",RList)
X=RList.reshape(RList.shape[0],21,21).astype('float32')
Y=NewLabel
x=np.ones([Quantity-SEQUENCE,SEQUENCE,21,21])
y=np.ones([Quantity-SEQUENCE,3])
for steps in range(Quantity-SEQUENCE):
    x[steps]=X[steps:steps+SEQUENCE]
    y[steps]=Y[steps+SEQUENCE]
print(x.shape)
print(y.shape)
print("Model Loading")
train_X, test_X, train_y, test_y = train_test_split(
     x, y, test_size=0.2, random_state=4)
model=Sequential()
model.add(ConvLSTM2D(filters=32,kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(ConvLSTM2D(filters=32,kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(21*21, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(3,activation='softmax'))
#print(model.summary())
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#epochs=訓練次數，batch_size=一次多少資料
train_history=model.fit(x=train_X, y=train_y,validation_split=0.2, epochs=EpochSize,batch_size=64)
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
#show_train_history(train_history,'acc','val_acc')
#show_train_history(train_history,'loss','val_loss')
scores=model.evaluate(test_X,test_y,batch_size=64)
print("Accuracy=",scores[1])
prediction=model.predict_classes(test_X)
print(prediction)
#model.save("D:/ZhuanTi/Models/CNN_12.h5")
TimeEnd=time.time()

