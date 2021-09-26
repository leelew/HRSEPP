# ==============================================================================
# train DL models in different 1x1 regions follow by several steps:
#
# author: Lu Li
# mail: lilu35@mail2.sysu.edu.cn
# ==============================================================================


from data.make_train_data import make_train_data
from data.make_inference_data import make_inference_data


# ------------------------------------------------------------------------------
# 1. train mode (re-train once a month)
# ------------------------------------------------------------------------------
make_train_data(begin_date='2015-03-31', end_date='2017-03-31', 
                lat_lower=22, lat_upper=33, 
                lon_left=110, lon_right=123,
                window_size=1)


# ------------------------------------------------------------------------------
# 2. inference mode (once a day)
# ------------------------------------------------------------------------------
make_inference_data(begin_date='2017-12-01', end_date='2017-12-10', 
                    lat_lower=22, lat_upper=33, 
                    lon_left=110, lon_right=123,
                    window_size=1)
