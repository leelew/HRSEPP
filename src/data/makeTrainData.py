from read_smap import prepare_SMAP
from read_cldas import prepare_CLDAS_forcing, prepare_CLDAS_model
from preprocess_xy import preprocess_CLDAS
from grids_match_xy import grid_match_Xy
from make_xy import make_input
import numpy as np


def make_train_data(raw_X_path, raw_y_path,
                    daily_X_path, daily_y_path,
                    begin_date, end_date,
                    lat_lower=-90, lat_upper=90, 
                    lon_left=-180, lon_right=180,
                    
                    len_input=10, 
                    len_output=1, 
                    window_size=7, 
                    use_lag_y=True):


    print('1')
    # ----------------------------------------------------#
    # 1. read SMAP and CLDAS and processing to daily data.#
    # ----------------------------------------------------#
    prepare_SMAP(input_path=raw_y_path,
                 out_path=daily_y_path,
                 begin_date=begin_date, 
                 end_date=end_date, 
                 lat_lower=lat_lower, 
                 lat_upper=lat_upper, 
                 lon_left=lon_left, 
                 lon_right=lon_right)

    prepare_CLDAS_forcing(input_path=raw_X_path,
                          out_path=daily_X_path,
                          begin_date=begin_date, 
                          end_date=end_date, 
                          lat_lower=lat_lower, 
                          lat_upper=lat_upper, 
                          lon_left=lon_left, 
                          lon_right=lon_right)

    # ----------------------------------------------------#
    # 2. preprocess CLDAS data
    # ----------------------------------------------------#
    preprocess_CLDAS(input_path=daily_X_path, 
                     out_path=daily_X_path, 
                     begin_date=begin_date, 
                     end_date=end_date)

    # ----------------------------------------------------#
    # 3. grid matching 
    # ----------------------------------------------------#    
    X, y = grid_match_Xy(X_path=daily_X_path, 
                         y_path=daily_y_path, 
                         begin_date=begin_date, 
                         end_date=end_date)

    # ----------------------------------------------------#
    # 4. make final inputs for DL
    # ----------------------------------------------------# 
    X_f = np.full((X.shape[0], len_input, X.shape[1], X.shape[2], X.shape[-1]), np.nan)
    y_f = np.full((X.shape[0], len_output, X.shape[1], X.shape[2], 1), np.nan)

    for i in np.arange(X.shape[1]):
        for j in np.arange(X.shape[2]):
            X_f[:, :, i,j, :], y_f[:, i,j, :] = make_input(
                                X[:, i, j, :], 
                                y[:, i, j, :], 
                                len_input=len_input, 
                                len_output=len_output, 
                                window_size=window_size, 
                                use_lag_y=use_lag_y)

    return X_f, y_f


if __name__ == "__main__":
    
    make_train_data(raw_X_path='/hard/lilu/CLDAS_FORCING/CLDAS_FORCING/', 
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
