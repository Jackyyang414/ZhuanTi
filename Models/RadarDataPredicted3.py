# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 21:27:50 2019

@author: jacky
"""

import numpy as np
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import time
Weight=100
Height=100
Frames=16
SEQUENCE=np.load('SequenceArray.npz')['SequenceArray']#載入矩陣
print(SEQUENCE[0])
print('Data loaded')
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
NUMBER=len(SEQUENCE)
SEQUENCE2=[]
for i in range(int(NUMBER/2)):
     SEQUENCE2.append(SEQUENCE[2 * i])
#Step = 3
SEQUENCE3 = []
for i in range(int(NUMBER / 3)):
    SEQUENCE3.append(SEQUENCE[3 * i])
#def GetSequence()
BASIC_SEQUENCE=np.zeros((NUMBER-Frames,Weight*Height))
NEXT_SEQUENCE=np.zeros((NUMBER-Frames,Frames*Weight))
for i in range(Frames):
    print(i)
    BASIC_SEQUENCE[:, i]=SEQUENCE[i:i+NUMBER-Frames]
    NEXT_SEQUENCE[:, i]=SEQUENCE[i+1:i+NUMBER-Frames+1]
train_X, test_X, train_y, test_y = train_test_split(BASIC_SEQUENCE[:10], NEXT_SEQUENCE[:10], test_size=0.2, random_state=4)
model=MultiOutputRegressor(LinearSVR(loss='mean_square_error', C=1.0))
model.fit(train_X,train_y)
score = model.score(test_X, test_y)
train_loss = mean_squared_error(train_X, test_y)
val_loss = mean_squared_error(test_X, test_y)
print("Score", score)
print("train loss: %.4f - val_loss: %.4f" % (train_loss, val_loss))
