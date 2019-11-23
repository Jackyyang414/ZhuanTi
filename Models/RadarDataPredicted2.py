# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 20:47:46 2019

@author: jacky
"""
import numpy as np
#from PIL import Image
import os 
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
import time
from keras import optimizers
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
gpu_options=tf.GPUOptions(allow_growth=True)
sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#config = tf.ConfigProto()  
#config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
#sess = tf.Session(config=config)
KTF.set_session(sess)
Weight=100
Height=100
Frames=16
SEQUENCE=np.load('D:/ZhuanTi/Models/SequenceArray.npz')['SequenceArray']#載入矩陣
print(SEQUENCE[0])
print('Data loaded')
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
NUMBER=len(SEQUENCE)
#Step=1
SEQUENCE=SEQUENCE.reshape(NUMBER,Weight,Height,1)
#Step=2
SEQUENCE2=[]
for i in range(int(NUMBER/2)):
     SEQUENCE2.append(SEQUENCE[2 * i])
#Step = 3
SEQUENCE3 = []
for i in range(int(NUMBER / 3)):
    SEQUENCE3.append(SEQUENCE[3 * i])
#def GetSequence()
SEQUENCE=SEQUENCE.reshape(NUMBER,Weight,Height,1)
BASIC_SEQUENCE=np.zeros((NUMBER-Frames,Frames,Weight,Height,1))
NEXT_SEQUENCE=np.zeros((NUMBER-Frames,Frames,Weight,Height,1))
for i in range(Frames):
    print(i)
    BASIC_SEQUENCE[:, i, :, :, :] = SEQUENCE[i:i+NUMBER-Frames]
    NEXT_SEQUENCE[:, i, :, :, :] = SEQUENCE[i+1:i+NUMBER-Frames+1]
# build model    
Seq=Sequential()
Seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),input_shape=(None,Weight,Height,1), padding='same', return_sequences=True))
Seq.add(BatchNormalization())
Seq.add(ConvLSTM2D(filters=60, kernel_size=(3, 3), padding='same', return_sequences=True))
Seq.add(BatchNormalization())
#Seq.add(ConvLSTM2D(filters=60, kernel_size=(3, 3), padding='same', return_sequences=True))
#Seq.add(BatchNormalization())
#Seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True))
#Seq.add(BatchNormalization())
Seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3), activation='sigmoid', padding='same', data_format='channels_last'))
Sgd=optimizers.SGD(lr=0.01, clipnorm=1)
Seq.compile(loss='mean_squared_error', optimizer='adadelta')
#epochs=訓練次數，batch_size=一次多少資料
Seq.fit(BASIC_SEQUENCE[:10], NEXT_SEQUENCE[:10], batch_size=4,
        epochs=100, validation_split=0.05)
Seq.save('PreDictModel.h5')
Which=224
Track=BASIC_SEQUENCE[Which][:12, ::, ::, ::]
for j in range(Frames+1):
    NewPos=Seq.predict(Track[np.newaxis, ::, ::, ::, ::])
    New=NewPos[::, -1, ::, ::, ::]
    Track=np.concatenate((Track,New), axis=0)
# And then compare the predictions
# to the ground truth
Track2=BASIC_SEQUENCE[Which][::, ::, ::, ::]
for i in range(Frames):
    fig=plt.figure(figsize=(10, 5))
    ax=fig.add_subplot(121)
    if i >= 8:
        ax.text(1, 3, 'Predictions !', fontsize=20, color='w')
    else:
        ax.text(1, 3, 'Inital trajectory', fontsize=20)
    Toplot=Track[i, ::, ::, 0]
    plt.imshow(Toplot, cmap='binary')
    ax=fig.add_subplot(122)
    plt.text(1, 3, 'Ground truth', fontsize=20)
    Toplot=Track2[i, ::, ::, 0]
    if i >= 8:
        Toplot=NEXT_SEQUENCE[Which][i - 1, ::, ::, 0]
    plt.imshow(Toplot, cmap='binary')
    plt.savefig('D:/ZhuanTi/RadarEchoDatabase/PredictedRadarImage/%i_Predict.png' % (i + 1))
    print("Done saving image No.%i"%(i+1))

