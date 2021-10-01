# ==============================================================================
# Read SMAP  and crop spatial dimension and integrate temporal dimension.
#
# author: Lu Li, 2021/09/27
# ==============================================================================


import datetime as dt
import glob
import os

import h5py
import netCDF4 as nc
import numpy as np

from data.utils import get_date_array


# ------------------------------------------------------------------------------
# 1. read single file of SMAP and crop spatial dimension
# ------------------------------------------------------------------------------
def read_single_smap(path,
                     lat_lower,
                     lat_upper,
                     lon_left,
                     lon_right):

    # get date for data
    # TODO: now must obey SMAP file rules, need to extent to other file name.
    yyyymmddhh = path.split('/')[-1].split('_')[4]
    yyyy = int(yyyymmddhh[0:4])
    mm = int(yyyymmddhh[4:6])
    dd = int(yyyymmddhh[6:8])
    hh = int(yyyymmddhh[9:11])
    date = dt.datetime(yyyy, mm, dd, hh)
    print('Now reading SMAP Level 4 Surface Soil Moisture for {}'.format(date))

    # handle for HDF file
    f = h5py.File(path, 'r')

    # read lat, lon
    lat_matrix_global = f['cell_lat'][:]
    lon_matrix_global = f['cell_lon'][:]

    lat = lat_matrix_global[:, 0]
    lon = lon_matrix_global[0, :]

    # crop regions according lat/lon
    lat_idx = np.where((lat > lat_lower) & (lat < lat_upper))[0]
    lon_idx = np.where((lon > lon_left) & (lon < lon_right))[0]

    # read surface soil moisture
    ssm = f['Geophysical_Data']['sm_surface'][lat_idx, :][:, lon_idx]
    ssm[ssm == -9999] = np.nan

    lat_matrix = lat_matrix_global[lat_idx][:, lon_idx]
    lon_matrix = lon_matrix_global[lat_idx][:, lon_idx]

    return ssm, lat_matrix, lon_matrix, \
        date, \
        lat_matrix.shape[0], lat_matrix.shape[1]


def read_daily_smap(input_path, begin_date, end_date):

    # get dates array according to begin/end dates
    dates = get_date_array(begin_date, end_date)

    # get shape
    filename = 'SMAP_L4_SSM_{year}{month:02}{day:02}.nc'.\
        format(year=dates[0].year,
               month=dates[0].month,
               day=dates[0].day)
    f = nc.Dataset(input_path + filename, 'r')
    Nlat, Nlon = f['ssm'][:].shape

    # read file
    # ---------
    data = np.full((len(dates), Nlat, Nlon, 1), np.nan)

    for i, date in enumerate(dates):

        # file name
        filename = 'SMAP_L4_SSM_{year}{month:02}{day:02}.nc'.\
            format(year=date.year,
                   month=date.month,
                   day=date.day)

        # handle for nc file
        f = nc.Dataset(input_path + filename, 'r')

        # read forcing
        data[i, :, :, 0] = f['ssm'][:]
        lat = f['latitude'][:]
        lon = f['longitude'][:]

    return data, lat, lon
