import datetime as dt

import netCDF4 as nc
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


def _check_x(inputs):

    # check inputs by following constrains
    # 1) if any feature of inputs is all NaN, then discard this inputs.
    # 2) if dimensions of inputs is not equal to 4, then raise error.

    if inputs.ndim == 4:
        Nt, Nlat, Nlon, Nf = inputs.shape

        #for i in range(Nf):
        pass
    else:
        raise TypeError('The dimension is not equal to 4')


# ------------------------------------------------------------------------------
# 2. Preprocessing data by interplot and normalization for training mode
#    Note: preprocessing process isn't applied on SMAP.
# ------------------------------------------------------------------------------
def preprocess_train_daily_data(inputs):

    # get shape
    Nt, Nlat, Nlon, Nf = inputs.shape

    # interplot and scale for each feature on each grid
    # -------------------------------------------------
    inputs_min = np.full((Nlat, Nlon, Nf), np.nan)
    inputs_max = np.full((Nlat, Nlon, Nf), np.nan)

    for i in np.arange(Nlat):
        for j in np.arange(Nlon):

            try:  #if np.isnan(inputs
                # interplot
                #imp = SimpleImputer(missing_values=np.nan, strategy='mean')
                imp = SimpleImputer(missing_values=np.nan,
                                    strategy='constant',
                                    fill_value=0)  # advised by Qingliang Li

                inputs[:, i, j, :] = imp.fit_transform(inputs[:, i, j, :])  #

                # min max scaler
                scaler = MinMaxScaler()
                inputs[:, i, j, :] = scaler.fit_transform(inputs[:, i, j, :])
                inputs_min[i, j, :] = scaler.data_min_
                inputs_max[i, j, :] = scaler.data_max_
            except:
                print('all data is nan')

    return inputs, inputs_min, inputs_max


def preprocess_test_daily_data(inputs, input_preprocess_path):

    # get shape
    Nt, Nlat, Nlon, Nf = inputs.shape

    # get min/max
    #TODO: give a more smart operation to read min/max matrixs
    f = nc.Dataset(input_preprocess_path + 'CLDAS_force_P_20150531.nc', 'r')
    min_, max_ = f['min'][:], f['max'][:]

    # preprocess according normalized parameters
    for i in np.arange(Nlat):
        for j in np.arange(Nlon):

            try:
                # interplot
                #imp = SimpleImputer(missing_values=np.nan, strategy='mean')
                imp = SimpleImputer(missing_values=np.nan,
                                    strategy='constant',
                                    fill_value=0)  # advised by Qingliang Li

                inputs[:, i, j, :] = imp.fit_transform(inputs[:, i, j, :])

                # min max scaler
                for m in np.arange(Nf):
                    inputs[:, i, j, m] = \
                        (inputs[:, i, j, m]-min_[i, j, m]) / \
                        (max_[i, j, m]-min_[i, j, m])
            except:
                print('all data is nan')

    return inputs, min_, max_
