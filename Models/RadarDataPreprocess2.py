# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 22:31:38 2019

@author: jacky
"""

import os
import pickle as pkl
import datetime
import pandas as pd
PreprocessedRadarPath="D:/ZhuanTi/RadarEchoDatabase/PreprocessedRadarValue/"
RadarFolder=["June","July","August"]
for Month in RadarFolder:
    FileList=os.listdir(PreprocessedRadarPath+Month+"/")
    Radar=[]
    Date=[]
    for RadarValue in FileList:
        if RadarValue.split(".")[-1]==".pickle":
            Data=RadarValue.split(".")[1]+RadarValue.split(".")[2]
            DataFormat="%Y%m%d%H%M"
            Data=Data.replace(".", "")
            Data=datetime.datetime.strptime(Data,DataFormat)
            Date.append(Data)
            content=None
            with open(PreprocessedRadarPath+'{}/{}'.format(Month,RadarValue),'rb') as File:
                content = pkl.load(File)
            Radar.append(content)
            print("Radar========>",Radar)
    Dataframe = {
        "Radar": Radar,
        "DateTime": Date
    }
    RadarDataFrame=pd.DataFrame(Dataframe)
    RadarDataFrame["DateTime"]=pd.to_datetime(RadarDataFrame['DateTime'])
    print(RadarDataFrame.head())
    print(RadarDataFrame.info())
    RadarDataFrame.to_pickle(PreprocessedRadarPath+'RadarData_{}.pickle'.format(Month))
     
        
    