# ==============================================================================
# train DL models in different 1x1 regions follow by several steps:
#
# author: Lu Li
# mail: lilu35@mail2.sysu.edu.cn
# ==============================================================================


import os

import tensorflow as tf

from data.make_inference_data import make_inference_data
from data.make_test_data import make_test_data
from data.make_train_data import make_train_data
from model.LSTM import LSTM
from trainer import keras_train, load_model

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:

    #tf.config.experimental.set_visible_devices(devices=gpus[1], device_type='GPU')
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)
    

def main(mode):


    if mode=='train':
        # ------------------------------------------------------------------------------
        # 1. train mode (re-train once a month)
        # ------------------------------------------------------------------------------
        X, y = make_train_data(
                raw_X_path='/hard/lilu/CLDAS_FORCING/CLDAS_FORCING/', 
                raw_y_path='/hard/lilu/SMAP_L4/SMAP_L4/',
                daily_X_path='/hard/lilu/CLDAS_FORCING/CLDAS_FORCING_DD/', 
                daily_y_path='/hard/lilu/SMAP_L4/SMAP_L4_DD/',
                begin_date='2015-03-31', end_date='2017-03-31',
                lat_lower=22, lat_upper=33, 
                lon_left=110, lon_right=123,
                
                len_input=5, 
                len_output=1, 
                window_size=3, 
                use_lag_y=True)

        model = LSTM(n_feature=X.shape[-1], input_len=5)

        keras_train(model, X, y, batch_size=128, epochs=10, save_folder='/hard/lilu/HRSEPP/')


    elif mode=='test':
        # ------------------------------------------------------------------------------
        # 2. inference mode (once a day)
        # ------------------------------------------------------------------------------

        X, y =make_test_data(
                raw_X_path='/hard/lilu/CLDAS_FORCING/CLDAS_FORCING/', 
                raw_y_path='/hard/lilu/SMAP_L4/SMAP_L4/',
                daily_X_path='/hard/lilu/CLDAS_FORCING/CLDAS_FORCING_DD/', 
                daily_y_path='/hard/lilu/SMAP_L4/SMAP_L4_DD/',
                begin_date='2017-04-01', end_date='2018-04-01',
                lat_lower=-90, lat_upper=90, 
                lon_left=-180, lon_right=180,
                    
                    len_input=10, 
                    len_output=1, 
                    window_size=7, 
                    use_lag_y=True)


        model = load_model()
        #inference(X,y)

        #postprocess()

    elif mode=='inference':
        # ------------------------------------------------------------------------------
        # 3. inference mode (once a day)
        # ------------------------------------------------------------------------------
        X = make_inference_data(begin_date='2017-12-01', end_date='2017-12-10', 
        #                    lat_lower=22, lat_upper=33, 
        #                    lon_left=110, lon_right=123,
        #                    window_size=1)
        model = load_model()
        y = model(X)

        

if __name__ == '__main__':
    main(mode='train')


