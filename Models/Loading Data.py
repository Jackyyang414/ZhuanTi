import numpy as np
import sys
import os
from random import randint
from sklearn.model_selection import train_test_split


class load_data(object):

    def __init__(self):
        self.file_list = os.listdir("D:\\ZhuanTi\\RadarEchoDatabase\\PreprocessedRadarValue\\June")
        self.n = len(self.file_list)
        self.height = 21
        self.width = 21
        self.feature = 21*self.height*self.width
        self.data = np.ones(shape=(self.n, 21, 21, 21))

        print("{} datas.".format(len(self.file_list)))
        for idx, rwd in enumerate(self.file_list):
            with open("D:\\ZhuanTi\\RadarEchoDatabase\\PreprocessedRadarValue\\June\\"+rwd) as file:
                content = np.loadtxt(file)
                self.data[idx] = content.reshape(-1, 21, 21)
        self.data = np.round(self.data, 3)
        print("Data shape ", self.data.shape)

    def load_data(self):
        return self.data

    def load_data_per(self, cnn, train_test, period=1):

        d = self.data.reshape(-1, self.feature)
        d_len = d.shape[0]
        X = np.empty([d_len-period, period, self.feature])
        y = np.empty([d_len-period, self.feature])

        for idx in range(d_len-period):
            X[idx] = d[idx:idx+period]
            y[idx] = d[idx+period]

        if cnn:
            X = X.reshape(-1, 5, 21, self.height, self.width)
            # X = X.transpose(0, 2, 3, 4, 1)
        

        train_X, test_X, train_y, test_y = train_test_split(
            X, y, test_size=train_test, random_state=4)

        print("Training set", train_X.shape[0], train_X.shape[1:])
        print("Testing set ", test_X.shape[0], train_X.shape[1:])

        return train_X, test_X, train_y, test_y

    def batch_generator(self, X, y, batch_size):

        d_len = y.shape[0]-batch_size
        while True:
            i = randint(0, d_len)
            temp_X = X[i:i+batch_size]
            temp_y = y[i:i+batch_size]
            yield(temp_X, temp_y)
        

    def batch_generator_tf(self, X, y, batch_size):

        d_len = y.shape[0]-batch_size
        
        i = randint(0, d_len)
        temp_X = X[i:i+batch_size]
        temp_y = y[i:i+batch_size]
        return temp_X, temp_y