

import glob
from data.make_train_data import make_train_data
from model.LSTM import LSTM

import numpy as np
import netCDF4 as nc
import os

def train_task(input_path, saved_model_path, task):

    """
    # make train data for predicting
    # ----------------------------------
    make_train_data(begin_date=begin_date, end_date=end_date, 
                    lat_lower=22, lat_upper=33, 
                    lon_left=110, lon_right=123,
                    window_size=1)
    """

    # 读取数据
    # -------
    filename = 'train_input_task_{}.nc'.format(task)
    f = nc.Dataset(input_path + filename, 'r')
    print(input_path + filename)
    x_train = f['x_train'][:]
    y_train = f['y_train'][:]

    # 构建模型
    # -------
    model = LSTM()

    if x_train.shape[0] != 0:

        # 训练模型
        # -------
        model.compile(optimizer='adam',loss='mse')
        model.fit(x_train, y_train, batch_size=5000)

    # 存储训练模型的结果
    # ---------------
    model.save(saved_model_path)

def train(input_path, saved_model_path):

    # 
    l = glob.glob(input_path + 'train*.nc', recursive=True)

    # 
    for task in np.arange(len(l)):
        if not os.path.exists(saved_model_path+str(task)+'/'):
            os.mkdir(saved_model_path+str(task)+'/')
        
        train_task(input_path, saved_model_path+str(task)+'/', task)

if __name__ == '__main__':
    train(input_path='/hard/lilu/4_HRSMPP/train/',
          saved_model_path='/hard/lilu/4_HRSMPP/saved_models/')

