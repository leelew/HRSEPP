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

from config import parse_args
from data.data_generator import DataGenerator, DataLoader
from trainer import train


def main(id):
    """Main process for no backbone model."""
    config = parse_args

    # Data generator
    # NOTE: This module generate train/valid/test data
    #       for SMAP l3/l4 data with shape as
    #       (samples, height, width, features).
    #       We do not make inputs at once because the 
    #       process of reading inputs is complete.
    if not os.path.exists('/hard/lilu/inputs/SMAP_L4/X_train_l4_{}.npy'.format(id)):

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
                      end_date='2021-10-28',
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

    # load land mask
    with open('/hard/lilu/SMAP_L4/test/auxiliary.json') as f:
        aux = json.load(f)
        mask = np.squeeze(np.array(aux['mask']))
        lat_id_low, lon_id_left = aux['lat_low'][id - 1], aux['lon_left'][id - 1]
        land_mask = mask[lon_id_left:lon_id_left + 112, lat_id_low:lat_id_low + 112]
    print(land_mask.shape)
    # load SMAP enhanced l3
    x_train_l3 = np.load('/hard/lilu/inputs/SMAP_E_L3/X_train_l3.npy')[:, :, lon_id_left:lon_id_left + 112, lat_id_low:lat_id_low + 112]
    x_valid_l3 = np.load('/hard/lilu/inputs/SMAP_E_L3/X_valid_l3.npy')[:, :, lon_id_left:lon_id_left + 112, lat_id_low:lat_id_low + 112]
    x_test_l3 = np.load('/hard/lilu/inputs/SMAP_E_L3/X_test_l3.npy')[:, :, lon_id_left:lon_id_left + 112, lat_id_low:lat_id_low + 112]

    # load SMAP L4
    x_train_l4 = np.load('/hard/lilu/inputs/SMAP_L4/X_train_l4_{}.npy'.format(id))
    x_valid_l4 = np.load('/hard/lilu/inputs/SMAP_L4/X_valid_l4_{}.npy'.format(id))
    x_test_l4 = np.load('/hard/lilu/inputs/SMAP_L4/X_test_l4_{}.npy'.format(id))

    y_train_l4 = np.load('/hard/lilu/inputs/SMAP_L4/y_train_l4_{}.npy'.format(id))
    y_valid_l4 = np.load('/hard/lilu/inputs/SMAP_L4/y_valid_l4_{}.npy'.format(id))
    y_test_l4 = np.load('/hard/lilu/inputs/SMAP_L4/y_test_l4_{}.npy'.format(id))

    
    dl = DataLoader(len_input=7, len_output=7, window_size=0, use_lag_y=True)
    X = [x_train_l4, x_valid_l4, x_test_l4, x_train_l3, x_valid_l3, x_test_l3]
    y = [y_train_l4, y_valid_l4, y_test_l4, None, None, None]
    X, y = dl(X, y)
    X_l3, X_l4, y = X[3:], X[:3], y[:3]
    print(len(X_l3))
    print(len(X_l4))
    print(len(y))


    if config.do_transfer_learning:
        #load era5
        x_train = np.load('/hard/lilu/inputs/ERA5/X_train_era5_{}.npy'.format(id))
        y_train = np.load('/hard/lilu/inputs/ERA5/y_train_era5_{}.npy'.format(id))

        # make input
        #FIXME: Add validate part in `DataLoader`.
        #       return nd.array if inputs is not a list.
        #       X, y = dl(x_train, y_train)
        X, y = dl([x_train], [y_train])
        X, y = X[0], y[0]

        # pretrain model
        




    # train
    train(X_l3, X_l4, y,
          land_mask,
          id,
          model_name='smnet',
          learning_rate=0.006051,
          n_filters_factor=1.051,
          filter_size=5,
          batch_size=2,
          epochs=50)


if __name__ == '__main__':
    main(1)
