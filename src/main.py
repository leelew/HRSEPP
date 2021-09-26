# ==============================================================================
# train DL models in different 1x1 regions follow by several steps:
#
# author: Lu Li
# mail: lilu35@mail2.sysu.edu.cn
# ==============================================================================


from trainer import trainer
from data.makeTrainData import make_train_data
from data.makeInferenceData import make_inference_data



def main(mode, begin_date, end_date):


    if mode=='train':
        # ------------------------------------------------------------------------------
        # 1. train mode (re-train once a month)
        # ------------------------------------------------------------------------------
        X, y = make_train_data(begin_date='2015-03-31', end_date='2017-03-31', 
                                lat_lower=22, lat_upper=33, 
                                lon_left=110, lon_right=123,
                                window_size=1)

        trainer(X,y)


    elif mode=='inference':
        # ------------------------------------------------------------------------------
        # 2. inference mode (once a day)
        # ------------------------------------------------------------------------------
        X, y = make_inference_data(begin_date='2017-12-01', end_date='2017-12-10', 
                            lat_lower=22, lat_upper=33, 
                            lon_left=110, lon_right=123,
                            window_size=1)

        inference(X,y)

        postprocess()


