


import datetime
import glob
import time

import netCDF4 as nc
import numpy as np
import tensorflow as tf

#from crawl import download_SMAP_L4_NRT
from data.make_inference_data import make_inference_data
from postprocess import postprocess


def inference_task(input_path, date, out_path, saved_model_path, task):

    """inference for target date"""

    """
    # download near-real-time SMAP L4/ CLDAS
    # -------------------------------
    download_SMAP_L4_NRT()

    # get begin/end date for predicting tomorrow soil moisture
    # --------------------------------------------------------

    # 当天的日期。
    local_time = time.localtime()
    year = local_time.tm_year
    month = local_time.tm_mon
    day = local_time.tm_mday

    # 获取3天前-13天前的起止日期。
    begin_date = datetime.datetime.now() - datetime.timedelta(days = 12)
    end_date = datetime.datetime.now() - datetime.timedelta(days = 3)

    # make inference data for predicting
    # ----------------------------------
    make_inference_data(begin_date=begin_date, end_date=end_date, 
                        lat_lower=22, lat_upper=33, 
                        lon_left=110, lon_right=123,
                        window_size=1)
    """


    # 获取3天前-13天前的起止日期。
    begin_date = date - datetime.timedelta(days = 12)
    end_date = date - datetime.timedelta(days = 3)    

    # make inference data for predicting
    # ----------------------------------
    make_inference_data(begin_date=begin_date, end_date=end_date, 
                        lat_lower=22, lat_upper=33, 
                        lon_left=110, lon_right=123,
                        window_size=1)

    # 加载每日的预测输入
    # ---------------
    filename = 'predict_task_{}.nc'.format(task)
    f = nc.Dataset(input_path + filename, 'r')
    print(input_path + filename)
    x_predict = f['x_predict'][:]
    lon_ = f['longitude'][:]
    lat_ = f['latitude'][:]

    print(np.isnan(x_predict).any())

    # get shape
    # ---------
    Nlat, Nlon, Nt, Nf = x_predict.shape

    # 加载存储的模型，不同区域的预测
    # -------------------------
    model = tf.keras.models.load_model(saved_model_path)


    y_predict = np.full((Nlat,Nlon), np.nan)
    for i in np.arange(Nlat):
        for j in np.arange(Nlon):
            if not np.isnan(x_predict[i,j,:,:]).any():
                
                y_predict[i,j] = model.predict(x_predict[i,j,:,:][np.newaxis,:,:])

    print(np.isnan(y_predict).any())

    # 拼接，输出预测的结果。
    # -----------------
    # save to nc files
    filename = 'output_task_{}.nc'.format(task)

    f = nc.Dataset(out_path + filename, 'w', format='NETCDF4')

    f.createDimension('longitude', size=lon_.shape[0])
    f.createDimension('latitude', size=lat_.shape[0])

    lon = f.createVariable('longitude', 'f4', dimensions='longitude')
    lat = f.createVariable('latitude', 'f4', dimensions='latitude')
    predict = f.createVariable('y_predict', 'f4', \
        dimensions=('latitude', 'longitude'))

    lon[:] = lon_
    lat[:] = lat_
    predict[:] = y_predict

    f.close()


def inference(input_path, out_path, saved_model_path, date):

    # 
    l = glob.glob(input_path + 'predict*.nc', recursive=True)

    # 
    for task in np.arange(len(l)):
        
        inference_task(input_path, date, out_path, saved_model_path+str(task)+'/', task)

    


if __name__ == '__main__':
    date = datetime.strptime('2017-04-01', '%Y-%m-%d')
    inference(input_path='/hard/lilu/4_HRSMPP/inference/',
              out_path='/hard/lilu/4_HRSMPP/predict/',
              saved_model_path='/hard/lilu/4_HRSMPP/saved_models/',
              date=date)
    
    postprocess(
          predict_path='/hard/lilu/4_HRSMPP/predict/', 
          out_path='/hard/lilu/4_HRSMPP/', 
                lat_lower=22, lat_upper=33, 
                lon_left=110, lon_right=123, window_size=1, date=date)

