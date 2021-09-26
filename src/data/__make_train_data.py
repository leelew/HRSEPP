# ==============================================================================
# train DL models in different 1x1 regions follow by several steps:
# 1. make train dataset
# author: Lu Li
# mail: lilu35@mail2.sysu.edu.cn
# ==============================================================================

from train import prepare_SMAP, prepare_CLDAS_forcing
from .preprocess import preprocess_CLDAS
from .slices import slices_inputs, make_train_input


def make_train_data(begin_date, end_date, 
                    lat_lower, lat_upper, 
                    lon_left, lon_right,
                    window_size):

    """
    # integrate into daily and crop spatial dimension
    # -----------------------------------------------
    prepare_SMAP(input_path='/hard/lilu/SMAP_L4/SMAP_L4/',
                 out_path='/hard/lilu/SMAP_L4/SMAP_L4_DD/',
                 begin_date=begin_date, end_date=end_date,
                 lat_lower=lat_lower, lat_upper=lat_upper,
                 lon_left=lon_left, lon_right=lon_right)
    
    prepare_CLDAS_forcing(input_path='/hard/lilu/CLDAS_FORCING/CLDAS_FORCING/',
                          out_path='/hard/lilu/CLDAS_FORCING/CLDAS_FORCING_DD/',
                          begin_date=begin_date, end_date=end_date,
                          lat_lower=lat_lower, lat_upper=lat_upper,
                          lon_left=lon_left, lon_right=lon_right)

    # preprocess CLDAS forcing
    # ------------------------
    preprocess_CLDAS(input_path='/hard/lilu/CLDAS_FORCING/CLDAS_FORCING_DD/', 
                     out_path='/hard/lilu/CLDAS_FORCING/CLDAS_FORCING_DD_PREP/', 
                     begin_date=begin_date, end_date=end_date)
    """
    # slice into 1x1 regions
    # ----------------------
    slices_inputs(input_SMAP_path='/hard/lilu/SMAP_L4/SMAP_L4_DD/',
                  input_CLDAS_path='/hard/lilu/CLDAS_FORCING/CLDAS_FORCING_DD_PREP/', 
                  out_path='/hard/lilu/4_HRSMPP/inputs/',
                  begin_date=begin_date, end_date=end_date,
                  lat_lower=lat_lower, lat_upper=lat_upper,
                  lon_left=lon_left, lon_right=lon_right, 
                  window_size=window_size)
    
    # make train data for DL models
    # -----------------------------
    make_train_input(input_path='/hard/lilu/4_HRSMPP/inputs/', 
                     out_path='/hard/lilu/4_HRSMPP/train/')

    

    