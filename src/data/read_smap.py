# ==============================================================================
# Read SMAP  and crop spatial dimension and integrate temporal dimension.
#
# author: Lu Li, 2021/09/27
# ==============================================================================

import datetime as dt

import h5py
import netCDF4 as nc
import numpy as np

from ..utils.time import get_date_array


def read_smap(path, lat_lower, lat_upper, lon_left, lon_right):
    """Read from single file of SMAP and crop spatial dimension.

    Args:
        path ([type]): [description]
        lat_lower ([type]): [description]
        lat_upper ([type]): [description]
        lon_left ([type]): [description]
        lon_right ([type]): [description]

    Returns:
        [type]: [description]
    """

    # get date for data TODO:Extent to other type of file name.
    yyyymmddhh = path.split('/')[-1].split('_')[4]
    yyyy = int(yyyymmddhh[0:4])
    mm = int(yyyymmddhh[4:6])
    dd = int(yyyymmddhh[6:8])
    hh = int(yyyymmddhh[9:11])
    date = dt.datetime(yyyy, mm, dd, hh)
    print('[HRSEPP][IO] Reading SMAP surface soil moisture data of {}'.format(
        date))

    # handle for HDF file
    f = h5py.File(path, 'r')

    # read attribute (lat, lon) and crop attributes
    lat, lon = f['cell_lat'][:, 0], f['cell_lon'][0, :]

    lat_idx = np.where((lat > lat_lower) & (lat < lat_upper))[0]
    lon_idx = np.where((lon > lon_left) & (lon < lon_right))[0]

    lat_2d = f['cell_lat'][lat_idx, :][:, lon_idx]
    lon_2d = f['cell_lon'][lat_idx, :][:, lon_idx]

    # read surface soil moisture and forcing of target regions
    ssm = f['Geophysical_Data']['sm_surface'][lat_idx, :][:, lon_idx]

    # turn fillvalue to NaN
    ssm[ssm == -9999] = np.nan

    return ssm, lat_2d, lon_2d


def read_single_smap_forcing(path, lat_lower, lat_upper, lon_left, lon_right):
    """Read from single file of SMAP and crop spatial dimension.

    Args:
        path ([type]): [description]
        lat_lower ([type]): [description]
        lat_upper ([type]): [description]
        lon_left ([type]): [description]
        lon_right ([type]): [description]

    Returns:
        [type]: [description]
    """

    # get date for data TODO:Extent to other type of file name.
    yyyymmddhh = path.split('/')[-1].split('_')[4]
    yyyy = int(yyyymmddhh[0:4])
    mm = int(yyyymmddhh[4:6])
    dd = int(yyyymmddhh[6:8])
    hh = int(yyyymmddhh[9:11])
    date = dt.datetime(yyyy, mm, dd, hh)
    print('[HRSEPP][IO] Reading SMAP forcing data of {}'.format(date))

    # handle for HDF file
    f = h5py.File(path, 'r')

    # read attribute (lat, lon) and crop attributes
    lat, lon = f['cell_lat'][:, 0], f['cell_lon'][0, :]

    lat_idx = np.where((lat > lat_lower) & (lat < lat_upper))[0]
    lon_idx = np.where((lon > lon_left) & (lon < lon_right))[0]

    lat_2d = f['cell_lat'][lat_idx, :][:, lon_idx]
    lon_2d = f['cell_lon'][lat_idx, :][:, lon_idx]

    # read surface soil moisture and forcing of target regions
    p = f['Geophysical_Data']['precipitation_total_surface_flux'][
        lat_idx, :][:, lon_idx][:, :, np.newaxis]
    lw = f['Geophysical_Data']['radiation_longwave_absorbed_flux'][
        lat_idx, :][:, lon_idx][:, :, np.newaxis]
    sw = f['Geophysical_Data']['radiation_shortwave_downward_flux'][
        lat_idx, :][:, lon_idx][:, :, np.newaxis]
    sh = f['Geophysical_Data']['specific_humidity_lowatmmodlay'][
        lat_idx, :][:, lon_idx][:, :, np.newaxis]
    sp = f['Geophysical_Data']['surface_pressure'][
        lat_idx, :][:, lon_idx][:, :, np.newaxis]
    st = f['Geophysical_Data']['surface_temp'][lat_idx, :][:,
                                                           lon_idx][:, :,
                                                                    np.newaxis]
    ws = f['Geophysical_Data']['windspeed_lowatmmodlay'][
        lat_idx, :][:, lon_idx][:, :, np.newaxis]
    force = np.concatenate([p, lw, sw, sh, sp, st, ws], axis=-1)

    # turn fillvalue to NaN
    force[force == -9999] = np.nan

    return force, lat_2d, lon_2d


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


def read_daily_smap_force(input_path, begin_date, end_date):

    # get dates array according to begin/end dates
    dates = get_date_array(begin_date, end_date)

    # get shape
    filename = 'SMAP_L4_force_{year}{month:02}{day:02}.nc'.\
        format(year=dates[0].year,
               month=dates[0].month,
               day=dates[0].day)
    f = nc.Dataset(input_path + filename, 'r')
    Nlat, Nlon, Nf = f['force'][:].shape

    # read file
    # ---------
    data = np.full((len(dates), Nlat, Nlon, Nf), np.nan)

    for i, date in enumerate(dates):

        # file name
        filename = 'SMAP_L4_force_{year}{month:02}{day:02}.nc'.\
            format(year=date.year,
                   month=date.month,
                   day=date.day)

        # handle for nc file
        f = nc.Dataset(input_path + filename, 'r')

        # read forcing
        data[i, :, :, :] = f['force'][:]
        lat = f['latitude'][:]
        lon = f['longitude'][:]

    return data, lat, lon


def read_preprocessed_daily_smap_force(input_path, begin_date, end_date):

    # get dates array according to begin/end dates
    dates = get_date_array(begin_date, end_date)

    # get shape
    filename = 'SMAP_L4_force_preprocessed_{year}{month:02}{day:02}.nc'.\
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
        filename = 'SMAP_L4_force_preprocessed_{year}{month:02}{day:02}.nc'.\
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
