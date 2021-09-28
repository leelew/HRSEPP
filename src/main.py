# ==============================================================================
# train DL models in different 1x1 regions follow by several steps:
#
# author: Lu Li
# mail: lilu35@mail2.sysu.edu.cn
# ==============================================================================


from data.makeTrainData import make_train_data
from trainer import keras_train
from model.LSTM import LSTM

#from data.makeInferenceData import make_inference_data



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


    elif mode=='inference':
        # ------------------------------------------------------------------------------
        # 2. inference mode (once a day)
        # ------------------------------------------------------------------------------
        #X, y = make_inference_data(begin_date='2017-12-01', end_date='2017-12-10', 
        #                    lat_lower=22, lat_upper=33, 
        #                    lon_left=110, lon_right=123,
        #                    window_size=1)

        #inference(X,y)

        #postprocess()
        pass

if __name__ == '__main__':
    main(mode='train')


