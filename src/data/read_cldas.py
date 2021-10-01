# ==============================================================================
# Read CLDAS and crop spatial dimension and integrate temporal dimension.
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
# 1. Read CLDAS forcing and crop spatial dimension
# ------------------------------------------------------------------------------
def read_single_cldas_forcing(path,
                              variable,
                              lat_lower,
                              lat_upper,
                              lon_left,
                              lon_right):
    # get date for data
    yyyymmddhh = path.split('-')[-1].split('.')[0]
    yyyy = int(yyyymmddhh[0:4])
    mm = int(yyyymmddhh[4:6])
    dd = int(yyyymmddhh[6:8])
    hh = int(yyyymmddhh[9:11])
    date = dt.datetime(yyyy, mm, dd, hh)
    print('Now reading CLDAS for {}'.format(date))

    # handle for HDF file
    f = h5py.File(path, 'r')

    # read lat, lon
    lat = f['LAT'][:]
    lon = f['LON'][:]

    # crop regions according lat/lon
    lat_idx = np.where((lat > lat_lower) & (lat < lat_upper))[0]
    lon_idx = np.where((lon > lon_left) & (lon < lon_right))[0]

    # read forcing
    data = f[variable][lat_idx, :][:, lon_idx]
    data[data < -99] = np.nan

    lat_region = lat[lat_idx]
    lon_region = lon[lon_idx]

    return data, lat_region, lon_region, date, \
        lat_region.shape[0], lon_region.shape[0]


# ------------------------------------------------------------------------------
# 1. Read CLDAS and SMAP daily data after applying prepare step
# ------------------------------------------------------------------------------
def read_daily_cldas_forcing(input_path, begin_date, end_date):

    # get dates array according to begin/end dates
    dates = get_date_array(begin_date, end_date)

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
        data[i, :, :, :] = f['forcing'][:]
        lat = f['latitude'][:]
        lon = f['longitude'][:]

    return data, lat, lon


# ------------------------------------------------------------------------------
# 3. Preprocessing data by interplot and normalization for inference mode
#    Note: preprocessing process isn't applied on SMAP.
# ------------------------------------------------------------------------------
def read_preprocessed_daily_cldas_forcing(input_path,
                                          begin_date,
                                          end_date):

    # get dates array according to begin/end dates
    dates = get_date_array(begin_date, end_date)

    # get shape
    filename = 'CLDAS_force_P_{year}{month:02}{day:02}.nc'.\
        format(year=dates[0].year,
               month=dates[0].month,
               day=dates[0].day)
    f = nc.Dataset(input_path + filename, 'r')
    Nlat, Nlon, Nf = f['forcing'][:].shape
    min_, max_ = f['min'][:], f['max'][:]

    # read file
    # ---------
    data = np.full((len(dates), Nlat, Nlon, Nf), np.nan)

    for i, date in enumerate(dates):

        # file name
        filename = 'CLDAS_force_P_{year}{month:02}{day:02}.nc'.\
            format(year=date.year,
                   month=date.month,
                   day=date.day)

        # handle for nc file
        f = nc.Dataset(input_path + filename, 'r')

        # read forcing
        data[i, :, :, :] = f['forcing'][:]
        lat = f['latitude'][:]
        lon = f['longitude'][:]

    return data, lat, lon, min_, max_
