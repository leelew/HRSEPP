# ==============================================================================
# Make input for single grid or multi-grids (images)
#
# author: Lu Li, 2021/09/27
# ============================================================================== 

import numpy as np

from data.grids_match_xy import grid_match_Xy
from data.make_xy import make_image_inputs
from data.preprocess_train_xy import preprocess_CLDAS
from data.read_cldas import prepare_CLDAS_forcing, prepare_CLDAS_model
from data.read_smap import prepare_SMAP


def make_train_data(raw_X_path, raw_y_path,
                    daily_X_path, daily_y_path,
                    begin_date, end_date,
                    lat_lower=-90, lat_upper=90, 
                    lon_left=-180, lon_right=180,
                    
                    len_input=10, 
                    len_output=1, 
                    window_size=7, 
                    use_lag_y=True):

    # ----------------------------------------------------#
    # 1. read SMAP and CLDAS and processing to daily data.#
    # ----------------------------------------------------#
    print('\033[1;31m%s\033[0m' % 'Read inputs, crop spatial dimensions, aggregate into daily scale')
    
    """
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
    """
    # ----------------------------------------------------#
    # 2. preprocess CLDAS data
    # ----------------------------------------------------#
    print('\033[1;31m%s\033[0m' % 'Preprocess inputs')

    """
    preprocess_CLDAS(input_path=daily_X_path, 
                     out_path=daily_X_path, 
                     begin_date=begin_date, 
                     end_date=end_date)
    """
    
    # ----------------------------------------------------#
    # 3. grid matching 
    # ----------------------------------------------------#  
    print('\033[1;31m%s\033[0m' % 'Grid matching')
  
    X, y = grid_match_Xy(X_path=daily_X_path, 
                         y_path=daily_y_path, 
                         begin_date=begin_date, 
                         end_date=end_date)

    
    # ----------------------------------------------------#
    # 4. make final inputs for DL
    # ----------------------------------------------------#
    print('\033[1;31m%s\033[0m' % 'Make inputs')

    X, y = make_image_inputs(X, 
                             y,
                             len_input=len_input, 
                             len_output=len_output, 
                             window_size=window_size, 
                             use_lag_y=use_lag_y)

    return X, y


if __name__ == "__main__":
    
    make_train_data(
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
