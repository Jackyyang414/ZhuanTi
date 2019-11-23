import os
import pandas as pd
import numpy as np
import datetime

month = ["JUN","JUL","AUG"]
for m in month:
    file_list = os.listdir("E:/2017_gauge/{}/".format(m))
    STID=[] 
    STUM=[]
    TIME=[]
    LAT=[] 
    LON=[]
    ELEV=[]
    RAIN=[]
    MIN_10=[]
    HOUR_3=[]
    HOUR_6=[]
    HOUR_12=[]
    HOUR_24=[]
    NOW=[]
    CITY=[]
    CITY_SN=[]
    TOWN=[]
    TOWN_SN=[]
    ATTRIBUTE=[]
    no=[]
    date = []
    for rwd in file_list:
        if rwd[-3:] == 'txt':
            dt = rwd[0:12]
            d_format = "%Y%m%d%H%M"
            dt = dt.replace(".", "")
            dt = datetime.datetime.strptime(dt, d_format)
            date.append(dt)
            with open('E:/2017_gauge/{}/{}'.format(m, rwd)) as File:
                content = File.readlines()[:-2]
                
            for line in content:
                data = line.split(',')[0]
                data = np.array(data.split(' '))
                #print(data)
                    
                STID.append(data[0])
                #print(STID)
                STUM.append(data[1])
                #print(STUM)
                TIME.append(data[2])
                LAT.append(data[3]) 
                LON.append(data[4])
                ELEV.append(data[5])
                RAIN.append(data[6])
                MIN_10.append(data[7])
                HOUR_3.append(data[8])
                HOUR_6.append(data[9])
                HOUR_12.append(data[10])
                HOUR_24.append(data[11])
                NOW.append(data[12])
                CITY.append(data[13])
                CITY_SN.append(data[14])
                TOWN.append(data[15])
                TOWN_SN.append(data[16])
                ATTRIBUTE.append(data[17])
            dataframe={
                        'STID':STID,
                        'STUM':STUM,
                        'TIME':TIME,
                        'LAT':LAT,
                        'LON':LON,
                        'ELEV':ELEV,
                        'RAIN':RAIN,
                        'MIN_10':MIN_10,
                        'HOUR_3':HOUR_3,
                        'HOUR_6':HOUR_6,
                        'HOUR_12':HOUR_12,
                        'HOUR_24':HOUR_24,
                        'NOW':NOW,
                        'CITY':CITY,
                        'CITY_SN':CITY_SN,
                        'TOWN':TOWN,
                        'TOWN_SN':TOWN_SN,
                        'ATTRIBUTE':ATTRIBUTE,
                        #'DateTime':date
                            }
            rainfall_df = pd.DataFrame(dataframe, dtype = 'float')
            rainfall_df = rainfall_df[rainfall_df['STID'] == 'C0F9N0']#大里
            rainfall_df.insert(0, 'DateTime', date)
            rainfall_df.set_index("DateTime")
            rainfall_df["DateTime"] = pd.to_datetime(rainfall_df['DateTime'])

rainfall_df=rainfall_df.interpolate()

rainfall_df.to_pickle('D:/ZhuanTi/RainDataBase/RainfallData_{}.pkl'.format(m))