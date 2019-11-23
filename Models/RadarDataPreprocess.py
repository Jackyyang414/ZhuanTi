# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 23:25:58 2019
ModelVersion1.0
@author: jacky
"""
import os
import numpy as np
import pickle as pkl
import time
PreprocessedRadarPath="D:/ZhuanTi/RadarEchoDatabase/PreprocessedRadarValue/"
RadarFileDir=["E:/Radar_Echo/Jun","E:/Radar_Echo/Jul","E:/Radar_Echo/Aug"]#六七八月雷達路徑
RadarFolder=["Jun","Jul","Aug"]
CenterX=216#大里區中心點x=為216
CenterY=328#大里區中心點y=328
RangeSize=21
x=CenterX-int(RangeSize/2)
y=CenterY-int(RangeSize/2)
Sum=0
Count=0
Missed=0
TimeStart=time.time()
print("RadarData Preprocessing........")
for Index,Files in enumerate(RadarFileDir):
    if not os.path.isdir(PreprocessedRadarPath+RadarFolder[Index]):
        os.mkdir(PreprocessedRadarPath+RadarFolder[Index])#創建處理過後的資料夾
    RadarList=os.listdir(Files)#未處理的資料
    ProcessedRadarList=os.listdir(PreprocessedRadarPath+RadarFolder[Index])#經處理的資料
    Total=len(RadarList)
    #print(RadarList)
    for RadarValue in RadarList:
        RadarData=[]
        PickledData=RadarValue[:-4]+'.pickle'#資料序列化(節省空間)
        if RadarValue[-4:]=='.txt' and PickledData not in ProcessedRadarList:
           Count+=1
           Sum+=1
           with open(Files+'/'+RadarValue) as file:
                Content=file.readlines()
                RadarData=Content[-1].replace('-999','0')#Content[-1]雷達迴波值
                RadarData=RadarData.replace('-99','0')
                try:
                    RadarData=np.array(RadarData.split('     ')).astype(np.float32)
                except:
                    with open(PreprocessedRadarPath+'MissingData.txt','a') as Missing:
                        Missing.write(Files+'/'+PickledData+'/n')
                        Sum+=1
                        Missed+=1
                    continue
           RadarData=RadarData.reshape(21,561,441)
           RadarData=RadarData[:,y:y+RangeSize,x:x+RangeSize]#?
           ProcessedRadarData=PreprocessedRadarPath+RadarFolder[Index]+"/P"+PickledData
           with open(ProcessedRadarData,'wb') as PRadarData:
                pkl.dump(RadarData,PRadarData)
        elif PickledData in ProcessedRadarList:
            Count+=1
    print("%d/%d Completed %3.2f％" % (Index+1,len(RadarFileDir),round(Count/Total*100, 2)))  
TimeEnd=time.time()
print("Finished!  {} files were processed, {} files were destroyed.\n".format(Sum,Missed))
print("It cost %f sec" % (TimeEnd-TimeStart))
                
           
            
        
    