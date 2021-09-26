
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



def grid_match(X, y, Xlat, Xlon, Xres, ylat, ylon, yres):

    if y.ndims == 3:
        N, Nlat, Nlon = y.shape

    matched_X = np.full((N, Nlat, Nlon, X.shape[-1]), np.nan)
    
    for i in range(len(ylat)):
        for j in range(len(ylon)):

            # grid match index
            lat, lon = ylat[i], ylon[j]
            lat_idx = np.where(Xlat < (lat + yres) & Xlat > (lat - yres))[0]
            lon_idx = np.where(Xlon < (lon + yres) & Xlon > (lon - yres))[0]

            # average mapping
            matched_X[:, i, j, :] = np.nanmean(
                X[:, lat_idx, lon_idx, :], axis=(-2,-3))

    return matched_X, y


def grid_match_Xy():
    pass