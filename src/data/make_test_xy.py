
from data.grid_match_xy import grid_match_xy
from data.make_grids_xy import make_grids_train_xy
from data.preprocess_x import (preprocess_raw_cldas_forcing,
                               preprocess_test_daily_cldas_forcing)
from data.preprocess_y import preprocess_raw_smap


def make_test_data(raw_X_path, raw_y_path,
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
    print('\033[1;31m%s\033[0m' %
          'Read inputs, crop spatial dimensions, aggregate into daily scale')

    preprocess_raw_smap(input_path=raw_y_path,
                        out_path=daily_y_path,
                        begin_date=begin_date,
                        end_date=end_date,
                        lat_lower=lat_lower,
                        lat_upper=lat_upper,
                        lon_left=lon_left,
                        lon_right=lon_right)

    preprocess_raw_cldas_forcing(input_path=raw_X_path,
                                 out_path=daily_X_path,
                                 begin_date=begin_date,
                                 end_date=end_date,
                                 lat_lower=lat_lower,
                                 lat_upper=lat_upper,
                                 lon_left=lon_left,
                                 lon_right=lon_right)

    # ----------------------------------------------------#
    # 2. preprocess CLDAS data according train data
    # ----------------------------------------------------#
    print('\033[1;31m%s\033[0m' % 'Preprocess inputs')

    preprocess_test_daily_cldas_forcing(input_path=daily_X_path,
                                        input_preprocess_path=daily_X_path,
                                        out_path=daily_X_path,
                                        begin_date=begin_date,
                                        end_date=end_date)

    # ----------------------------------------------------#
    # 3. grid matching
    # ----------------------------------------------------#
    print('\033[1;31m%s\033[0m' % 'Grid matching')

    X, y = grid_match_xy(X_path=daily_X_path,
                         y_path=daily_y_path,
                         begin_date=begin_date,
                         end_date=end_date)

    # ----------------------------------------------------#
    # 4. make final inputs for DL
    # ----------------------------------------------------#
    print('\033[1;31m%s\033[0m' % 'Make inputs')
    """
    X, y = make_grids_train_xy(X,
                               y,
                               len_input=len_input,
                               len_output=len_output,
                               window_size=window_size,
                               use_lag_y=use_lag_y)
    """
    return X, y
