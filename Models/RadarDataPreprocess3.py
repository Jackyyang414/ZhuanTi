import os
import sys
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import pandas as pd
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter, LinearLocator
from mpl_toolkits.mplot3d import Axes3D
# file_list = os.listdir(os.getcwd()+"\\original")
original_dir=["E:/Radar_Echo/Jun","E:/Radar_Echo/Jul","E:/Radar_Echo/Aug"]#六七八月雷達路徑
target_folder=["Jun","Jul","Aug"]
target_dir = 'D:/ZhuanTi/RadarEchoDatabase/PreprocessedRadarValue/'
# Dali Center  (X: 216, Y:328)
x=216
y=328
range_size=21
totals = 0
destroy = 0
tStart = time.time()
print("Processing....")
for idx, disk in enumerate(original_dir):
    count = 0
    if not os.path.isdir(target_dir+target_folder[idx]):
        os.mkdir(target_dir+target_folder[idx])

    file_list = os.listdir(disk)
    # target = os.listdir(os.getcwd()+"\\processed\\21x21\\"+target_folder[idx])
    if not os.path.isdir(target_dir+target_folder[idx]):
        os.mkdir(target_dir+target_folder[idx])
    target = os.listdir(target_dir+target_folder[idx])
    month_data = []
    datetimes = []
    total = len(file_list)
    for rwd in file_list:
        data = []
        print("%d/%d Completed %3.2f％" %
              (idx+1, len(original_dir), round(count/total*100, 2)))
        o = rwd[:-4] + '.pkl'
        m = rwd.split('.')[1:3]
        m = m[0] + m[1]
        d_format = "%Y%m%d%H%M"
        dt = datetime.datetime.strptime(m, d_format)
    

        # if rwd[-4:] == '.txt' and o not in target:
        count += 1
        totals += 1
        with open(disk+"\\"+rwd) as file:
            content = file.readlines()
            data = content[-1].replace('-999', '0')
            data = data.replace('-99', '0')
            try:
                data = np.array(data.split('     ')).astype(np.float32)
            except:
                with open(target_dir + 'Destroy_file.txt', 'a') as log:
                    log.write(disk+"\\"+o+'\n')
                    totals -= 1
                    destroy += 1
                continue

        data = data.reshape(21, 561, 441)
        data = data[:, y:y+range_size, x:x+range_size]
        output_n = target_dir + target_folder[idx] + "\\" + o
        datetimes.append(dt)
        month_data.append(data)
            # with open(output_n, 'wb') as output:
            #     pkl.dump(data, output)
    df = pd.DataFrame({
        "DateTime": datetimes,
        "Radar": month_data
    })
    df.to_pickle(target_dir+"Radar_ZhuanTi_21x21.pkl")

#         elif o in target:
#             count += 1
#     print("%d/%d Completed %3.2f％" %
#           (idx+1, len(original_dir), round(count/total*100, 2)))

# tEnd = time.time()
# print("Finished!  {} files were processed, {} files were destroyed.\n".format(totals, destroy))
# print("It cost %f sec" % (tEnd - tStart))
