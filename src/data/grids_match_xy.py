# ==============================================================================
# Read SMAP and CLDAS, and grid match inputs and outputs.
#
# author: Lu Li, 2021/09/27
# ============================================================================== 


import netCDF4 as nc
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

from utils import _get_date_array


def read_daily_SMAP(input_path,begin_date, end_date):

    # get dates array according to begin/end dates
    dates = _get_date_array(begin_date, end_date)

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
        data[i,:,:,0] = f['ssm'][:]
        lat = f['latitude'][:]
        lon = f['longitude'][:]

    return data, lat, lon

# ------------------------------------------------------------------------------
# 3. Preprocessing data by interplot and normalization for inference mode
#    Note: preprocessing process isn't applied on SMAP.
# ------------------------------------------------------------------------------
def read_p_daily_CLDAS_forcing(input_path, begin_date, end_date):

    # get dates array according to begin/end dates
    dates = _get_date_array(begin_date, end_date)

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
        data[i,:,:,:] = f['forcing'][:]
        lat = f['latitude'][:]
        lon = f['longitude'][:]

    return data, lat, lon, min_, max_


def grid_match(X, y, Xlat, Xlon, Xres, ylat, ylon, yres):

    if y.ndim != 4 & X.ndim != 4:
        raise ValueError("must 4D")
    else:
        N, Nlat, Nlon, _ = y.shape

    matched_X = np.full((N, Nlat, Nlon, X.shape[-1]), np.nan)
    
    for i in range(len(ylat)):
        for j in range(len(ylon)):

            # grid match index
            lat, lon = ylat[i], ylon[j]
            lat_idx = np.where((Xlat < (lat + yres/2)) & (Xlat > (lat - yres/2)))[0]
            lon_idx = np.where((Xlon < (lon + yres/2)) & (Xlon > (lon - yres/2)))[0]

            # average mapping
            matched_X[:, i, j, :] = np.nanmean(
                X[:, lat_idx, lon_idx, :], axis=(-2,-3))

    return matched_X, y


def grid_match_Xy(X_path, y_path, begin_date, end_date):
    
    y, ylat, ylon = read_daily_SMAP(input_path=y_path, 
                                    begin_date=begin_date, 
                                    end_date=end_date)

    X, Xlat, Xlon, _, _ = read_p_daily_CLDAS_forcing(
                                    input_path=X_path, 
                                    begin_date=begin_date, 
                                    end_date=end_date)

    assert y.shape[0] == X.shape[0]

    X, y = grid_match(X, y, Xlat, Xlon, 0.0625, ylat, ylon, 0.09) # 9km, 0.0625

    return X, y

    