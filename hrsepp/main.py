# ==============================================================================
# train DL models
#
# author: Lu Li
# mail: lilu35@mail2.sysu.edu.cn
# ==============================================================================
# TODO: split train/test/inference data, give a parameter
# to control restart train or test or inference

import json
import os

import numpy as np
import tensorflow as tf

from utils.config import parse_args

from data.data_generator import DataGenerator
from data.data_loader import DataLoader
from IO.train import train


def main(id):
    """Main process for no backbone model."""
    config = parse_args

    # make train, validate and test data
    DataGenerator(ID=id,
                  raw_data_path='/hard/lilu/SMAP_L4/SMAP_L4/',
                  auxiliary_data_path='/hard/lilu/SMAP_L4/test/',
                  save_x_path='/hard/lilu/SMAP_L4/test/forcing/',
                  save_y_path='/hard/lilu/SMAP_L4/test/SSM/',
                  save_px_path='/hard/lilu/SMAP_L4/test/px/',
                  save_py_path='/hard/lilu/SMAP_L4/test/py/',
                  lat_lower=14.7,
                  lat_upper=53.5,
                  lon_left=72.3,
                  lon_right=135,
                  begin_date='2015-05-31',
                  end_date='2019-05-31',
                  x_var_name='forcing',
                  x_var_list=[
                      'precipitation_total_surface_flux',
                      'radiation_longwave_absorbed_flux',
                      'radiation_shortwave_downward_flux',
                      'specific_humidity_lowatmmodlay', 'surface_pressure',
                      'surface_temp', 'windspeed_lowatmmodlay'
                  ],
                  y_var_name='SSM',
                  y_var_list=['sm_surface'],
                  mode='train',
                  save=True)()

    DataGenerator(ID=id,
                  raw_data_path='/hard/lilu/SMAP_L4/SMAP_L4/',
                  auxiliary_data_path='/hard/lilu/SMAP_L4/test/',
                  save_x_path='/hard/lilu/SMAP_L4/test/forcing/',
                  save_y_path='/hard/lilu/SMAP_L4/test/SSM/',
                  save_px_path='/hard/lilu/SMAP_L4/test/px/',
                  save_py_path='/hard/lilu/SMAP_L4/test/py/',
                  lat_lower=14.7,
                  lat_upper=53.5,
                  lon_left=72.3,
                  lon_right=135,
                  begin_date='2019-05-31',
                  end_date='2020-05-31',
                  x_var_name='forcing',
                  x_var_list=[
                      'precipitation_total_surface_flux',
                      'radiation_longwave_absorbed_flux',
                      'radiation_shortwave_downward_flux',
                      'specific_humidity_lowatmmodlay', 'surface_pressure',
                      'surface_temp', 'windspeed_lowatmmodlay'
                  ],
                  y_var_name='SSM',
                  y_var_list=['sm_surface'],
                  mode='valid',
                  save=True)()

    DataGenerator(ID=id,
                  raw_data_path='/hard/lilu/SMAP_L4/SMAP_L4/',
                  auxiliary_data_path='/hard/lilu/SMAP_L4/test/',
                  save_x_path='/hard/lilu/SMAP_L4/test/forcing/',
                  save_y_path='/hard/lilu/SMAP_L4/test/SSM/',
                  save_px_path='/hard/lilu/SMAP_L4/test/px/',
                  save_py_path='/hard/lilu/SMAP_L4/test/py/',
                  lat_lower=14.7,
                  lat_upper=53.5,
                  lon_left=72.3,
                  lon_right=135,
                  begin_date='2020-05-31',
                  end_date='2021-10-29',
                  x_var_name='forcing',
                  x_var_list=[
                      'precipitation_total_surface_flux',
                      'radiation_longwave_absorbed_flux',
                      'radiation_shortwave_downward_flux',
                      'specific_humidity_lowatmmodlay', 'surface_pressure',
                      'surface_temp', 'windspeed_lowatmmodlay'
                  ],
                  y_var_name='SSM',
                  y_var_list=['sm_surface'],
                  mode='test',
                  save=True)()

    # make inputs shape as [(s, t_in, lat, lon, f), (s, t_out, lat, lon, 1)]
    x_train = np.load('/hard/lilu/x_train_{}.npy'.format(id))
    y_train = np.load('/hard/lilu/y_train_{}.npy'.format(id))

    x_valid = np.load('/hard/lilu/x_valid_{}.npy'.format(id))
    y_valid = np.load('/hard/lilu/y_valid_{}.npy'.format(id))

    x_test = np.load('/hard/lilu/x_test_{}.npy'.format(id))
    y_test = np.load('/hard/lilu/y_test_{}.npy'.format(id))

    x_train, y_train = DataLoader(len_input=1,
                                  len_output=1,
                                  window_size=2,
                                  use_lag_y=True,
                                  mode='train')(x_train, y_train)

    x_valid, y_valid = DataLoader(len_input=1,
                                  len_output=1,
                                  window_size=2,
                                  use_lag_y=True,
                                  mode='valid')(x_valid, y_valid)

    x_test, y_test = DataLoader(len_input=1,
                                len_output=1,
                                window_size=2,
                                use_lag_y=True,
                                mode='valid')(x_test, y_test)

    # train
    with open('/hard/lilu/SMAP_L4/test/auxiliary.json') as f:
        aux = json.load(f)

    lat_id_low = aux['lat_low'][id - 1]
    lon_id_left = aux['lon_left'][id - 1]

    mask = np.array(aux['mask'])[lon_id_left:lon_id_left + 224,
                                 lat_id_low:lat_id_low + 224]

    train(x_train,
          y_train,
          x_valid,
          y_valid,
          mask,
          ID=id,
          input_shape=(224, 224, 8),
          learning_rate=0.001,
          n_filters_factor=1,
          filter_size=3,
          batch_size=16,
          epochs=50,
          n_forecast_months=1,
          n_output_classes=1,
          model_name='ed_convlstm',
          save_path='/hard/lilu/SMAP_L4/model/')


if __name__ == '__main__':
    main(mode='test')
    """GPU setting module
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')

    for gpu in gpus:
        tf.config.experimental.set_visible_devices(
            devices=gpus[1], device_type='GPU')
        tf.config.experimental.set_memory_growth(
            device=gpu, enable=True)
    """
