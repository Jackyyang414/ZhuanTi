import numpy as np
import pandas as pd
Path="E:/2017_gauge/JUN/"
File=open(Path+'201706010100.QPESUMS_GAUGE.10M.txt')
content=File.readlines()[:-2]
STID=[]
MIN_10=[]
for line in content:
    line=line.replace('-999.00','0.00')
    line=line.replace('-998.00','0.00')
    if(line[0:6]=='C0F9N0'):
       line.split(" ")
       print(float(line[49:53]))
    '''
    data = line.split(',')[0]
    data = np.array(data.split(' '))
    STID.append(data[0])
    MIN_10.append(data[7])
    dataframe={
        'STID':STID,
        #'MIN_10':MIN_10
    }
    rainfall_df = pd.DataFrame(dataframe, dtype = 'float')
    rainfall_df = rainfall_df[rainfall_df['STID'] == 'C0F9N0']
    #rainfall_df=rainfall_df.interpolate()
#print(MIN_10)
#print(content)
#print(dataframe)
    print(line[0:6])
'''