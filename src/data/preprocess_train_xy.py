# ==============================================================================
# Read daily CLDAS and preprocess CLDAS.
#
# author: Lu Li, 2021/09/27
# ============================================================================== 


import os

import netCDF4 as nc
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

from utils import _get_date_array


# ------------------------------------------------------------------------------
# 1. Read CLDAS and SMAP daily data after applying prepare step
# ------------------------------------------------------------------------------
def read_daily_CLDAS_forcing(input_path, begin_date, end_date):

    # get dates array according to begin/end dates
    dates = _get_date_array(begin_date, end_date)

    # get shape
    filename = 'CLDAS_force_{year}{month:02}{day:02}.nc'.\
        format(year=dates[0].year,
                month=dates[0].month,
                day=dates[0].day)
    f = nc.Dataset(input_path + filename, 'r')
    Nlat, Nlon, Nf = f['forcing'][:].shape

    # read file
    # ---------
    data = np.full((len(dates), Nlat, Nlon, Nf), np.nan)

    for i, date in enumerate(dates):
        
        # file name
        filename = 'CLDAS_force_{year}{month:02}{day:02}.nc'.\
            format(year=date.year,
                   month=date.month,
                   day=date.day)

        # handle for nc file
        f = nc.Dataset(input_path + filename, 'r')

        # read forcing
        data[i,:,:,:] = f['forcing'][:]
        lat = f['latitude'][:]
        lon = f['longitude'][:]

    return data, lat, lon


# ------------------------------------------------------------------------------
# 2. Preprocessing data by interplot and normalization for training mode
#    Note: preprocessing process isn't applied on SMAP.
# ------------------------------------------------------------------------------
def preprocess(inputs):

    # get shape
    Nt, Nlat, Nlon, Nf = inputs.shape

    # interplot and scale for each feature on each grid
    # -------------------------------------------------
    inputs_min = np.full((Nlat, Nlon, Nf), np.nan)
    inputs_max = np.full((Nlat, Nlon, Nf), np.nan)

    for i in np.arange(Nlat):
        for j in np.arange(Nlon):

            try:
                # interplot
                imp = SimpleImputer(missing_values=np.nan, strategy='mean')
                inputs[:,i,j,:] = imp.fit_transform(inputs[:,i,j,:])

                # min max scaler
                scaler = MinMaxScaler()
                inputs[:,i,j,:] = scaler.fit_transform(inputs[:,i,j,:])
                inputs_min[i,j,:] = scaler.data_min_
                inputs_max[i,j,:] = scaler.data_max_
            except:
                print('all data is nan')

    return inputs, inputs_min, inputs_max


def preprocess_CLDAS(input_path, out_path, begin_date, end_date):

    # read CLDAS and preprocess
    CLDAS, lat_, lon_ = read_daily_CLDAS_forcing(input_path, 
                                                 begin_date, end_date)
    CLDAS, min_, max_ = preprocess(CLDAS)

    # get dates array according to begin/end dates
    dates = _get_date_array(begin_date, end_date)

    # save file (after integrate spatial/temporal dimension)
    # ------------------------------------------------------
    for i, date in enumerate(dates):

        # save to nc files
        filename = 'CLDAS_force_P_{year}{month:02}{day:02}.nc'.\
            format(year=date.year,
                   month=date.month,
                   day=date.day)
        print('now we saving {}'.format(filename))

        f = nc.Dataset(out_path + filename, 'w', format='NETCDF4')

        f.createDimension('longitude', size=lon_.shape[0])
        f.createDimension('latitude', size=lat_.shape[0])
        f.createDimension('feature', size=CLDAS.shape[-1])

        lon = f.createVariable('longitude', 'f4', dimensions='longitude')
        lat = f.createVariable('latitude', 'f4', dimensions='latitude')
        forcing = f.createVariable('forcing', 'f4', \
            dimensions=('latitude', 'longitude', 'feature'))
        min = f.createVariable('min', 'f4', \
            dimensions=('latitude', 'longitude', 'feature'))
        max = f.createVariable('max', 'f4', \
            dimensions=('latitude', 'longitude','feature'))

        lon[:] = lon_
        lat[:] = lat_
        forcing[:] = CLDAS[i]
        min[:] = min_
        max[:] = max_
        
        f.close()
