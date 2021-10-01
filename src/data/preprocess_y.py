import glob
import os

import netCDF4 as nc
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

from data.read_smap import read_single_smap
from data.utils import get_date_array


# ------------------------------------------------------------------------------
# 1. read multiple files of SMAP according begin/end dates and integrate time
# ------------------------------------------------------------------------------
def preprocess_raw_smap(input_path,
                        out_path,
                        begin_date,
                        end_date,

                        # TODO: only support -90-90, -180-180!
                        lat_lower=-90,
                        lat_upper=90,
                        lon_left=-180,
                        lon_right=180):

    # get dates array according to begin/end dates
    dates = get_date_array(begin_date, end_date)

    # read and save file (after integrate spatial/temporal dimension)
    # ---------------------------------------------------------------
    for date in dates:

        # folder name
        foldername = '{year}.{month:02}.{day:02}/'.\
            format(year=date.year,
                   month=date.month,
                   day=date.day)

        # file list in each folder
        l = glob.glob(input_path + foldername + 'SMAP_L4*.h5', recursive=True)

        # save to nc files
        filename = 'SMAP_L4_SSM_{year}{month:02}{day:02}.nc'.\
            format(year=date.year,
                   month=date.month,
                   day=date.day)

        # judge already exist file
        if os.path.exists(out_path + filename):
            print('SMAP of {} already exist'.format(date))
        else:

            # get shape
            _, _, _, _, Nlat, Nlon = read_single_smap(l[0],
                                                      lat_lower, lat_upper,
                                                      lon_left, lon_right)

            # integrate from 3-hour to daily
            ssm_dd = np.full((Nlat, Nlon, len(l)), np.nan)

            for i, path in enumerate(l):
                ssm_dd[:, :, i], lat_matrix, lon_matrix, _, _, _ = \
                    read_single_smap(path,
                                     lat_lower, lat_upper,
                                     lon_left, lon_right)

            ssm_dd = np.nanmean(ssm_dd, axis=-1)

            # save to nc
            f = nc.Dataset(out_path + filename, 'w', format='NETCDF4')

            f.createDimension('longitude', size=Nlon)
            f.createDimension('latitude', size=Nlat)

            lon = f.createVariable('longitude', 'f4', dimensions='longitude')
            lat = f.createVariable('latitude', 'f4', dimensions='latitude')
            ssm = f.createVariable(
                'ssm', 'f4', dimensions=('latitude', 'longitude'))

            lon[:] = lon_matrix[0, :]
            lat[:] = lat_matrix[:, 0]
            ssm[:] = ssm_dd

            f.close()
