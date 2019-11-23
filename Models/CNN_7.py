import numpy as np
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.activations import relu
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import os
import matplotlib.pyplot as plt
import time
EpochSize=50
#EpochSize=5000
TimeStart=time.time()
np.random.seed(10)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#選定跑模型的GPU
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'#顯示所有訊息
#gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
gpu_options=tf.GPUOptions(allow_growth=True)#允許自動增加GPU的使用空間
sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
KTF.set_session(sess)
RainPath=["F:/fill_zero/RAINFALL_taichung/JUN","F:/fill_zero/RAINFALL_taichung/JUL","F:/fill_zero/RAINFALL_taichung/AUG"]#六七八月雨量路徑
RadarPath=["D:\\ZhuanTi\\RadarEchoDatabase\\PreprocessedRadarValue\\Jun","D:\\ZhuanTi\\RadarEchoDatabase\\PreprocessedRadarValue\\Jul","D:\\ZhuanTi\\RadarEchoDatabase\\PreprocessedRadarValue\\Aug"]#六七八月雷達路徑
Month=["June","July","August"]
DateFormat="%Y%m%d%H%M"
Keyword="C0F9N0"#大里雨量站代碼
Quantity=4320+4462+4464-27-12#(6月+7月+8月-missing data)
LabelQuantity=Quantity+12
RList=np.zeros(([Quantity,21,21,21])) 
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
        RList[a]=RadarArray
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
        content=RainValue.readlines()
        for j in range(350,450):#大里那行大概位於這個區間
            if content[j][0:6]==Keyword:
                Dali=content[j][35:69]#雨量部分
                RainTime=RainFile[0:12]#雨量時間(檔案名稱前部分)
                RainTimeList.append(RainTime)
                DaliRain=Dali.split(" ",6)
                if len(DaliRain)>6:
                   DaliRain[-1]='0.00'
                for index in range(len(DaliRain)):
                    DaliRain[index]=float(DaliRain[index])
                TotalRain=sum(DaliRain)
                if 0.34>TotalRain>=0.0:#Label 0=無雨 1=小雨 2=大雨
                    RainLabel=[1,0,0]
                    np.append(Label,RainLabel)
                if 6.6>=TotalRain>=0.34:
                    RainLabel=[0,1,0]
                    np.append(Label,RainLabel)
                if TotalRain>6.6:
                    RainLabel=[0,0,1]
                    np.append(Label,RainLabel)      
print("Finish Labeling")
DeleteLabelIndex=[0,1,2,3,4,5,6,7,8,9,10,11]
NewLabel=np.delete(Label,DeleteLabelIndex,axis=0)
np.savez("D:/ZhuanTi/RainDataBase/RainLabel.npz",NewLabel)
np.savez("D:\ZhuanTi\RadarEchoDatabase\PreprocessedRadarValue",RList)
X=RList.reshape(RList.shape[0],21,21,21).astype('float32')
Y=NewLabel
print(X.shape)
print(Y.shape)
train_X, test_X, train_y, test_y = train_test_split(
     X, Y, test_size=0.2, random_state=4)
#CNN Model
model = Sequential()
#Conv1
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', input_shape=(21,21,21), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#Conv2
model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.25))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
'''
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
'''
#model.add(BatchNormalization(axis=3))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(21*21*21, activation='relu'))
model.add(Dropout(0.25))
#輸出層
model.add(Dense(3,activation='softmax'))
#定義訓練方法
print(model.summary())
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
show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')
scores=model.evaluate(test_X,test_y,batch_size=64)
print("Accuracy=",scores[1])
prediction=model.predict_classes(test_X)
print(type(prediction))
print(prediction)
model.save("D:/ZhuanTi/Models/CNN_7.h5")
TimeEnd=time.time()
print("Finish!!It cost %f sec" % (TimeEnd-TimeStart))