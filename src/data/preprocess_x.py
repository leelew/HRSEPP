import glob
import os

import netCDF4 as nc
import numpy as np

from data.read_cldas import (read_daily_cldas_forcing,
                             read_preprocessed_daily_cldas_forcing,
                             read_single_cldas_forcing)
from data.utils import get_date_array, preprocess_daily_data


# ------------------------------------------------------------------------------
# 2. Read multiple CLDAS forcing and integrate temporal dimension
# ------------------------------------------------------------------------------
def preprocess_raw_cldas_forcing(input_path,
                                 out_path,
                                 begin_date,
                                 end_date,
                                 lat_lower=-90,
                                 lat_upper=90,
                                 lon_left=-180,
                                 lon_right=180):

    # get dates array according to begin/end dates
    dates = get_date_array(begin_date, end_date)

    # read and save file (after integrate spatial/temporal dimension)
    # ---------------------------------------------------------------
    variables = ['PRCP', 'PAIR', 'QAIR', 'SWDN', 'TAIR', 'WIND']
    forcing_name = ['PRE', 'PRS', 'SHU', 'SSRA', 'TMP', 'WIN']

    for date in dates:

        # folder name
        foldername = '{year}.{month:02}.{day:02}/'.\
            format(year=date.year,
                   month=date.month,
                   day=date.day)

        # file list in each folder
        l = glob.glob(input_path + foldername + '*PRE*.nc', recursive=True)

        filename = 'CLDAS_force_{year}{month:02}{day:02}.nc'.\
            format(year=date.year,
                   month=date.month,
                   day=date.day)

        if os.path.exists(out_path + filename):
            print("CLDAS on {} already exists".format(date))

        else:
            # get shape
            _, LAT, LON, _, Nlat, Nlon = read_single_cldas_forcing(l[0],
                                                                   'PRCP',
                                                                   lat_lower, lat_upper,
                                                                   lon_left, lon_right)

            # integrate from 3-hour to daily
            feature_dd = np.full((Nlat, Nlon, len(variables), len(l)), np.nan)

            for j, variable in enumerate(variables):

                # file list in each folder
                l = glob.glob(input_path +
                              foldername + '*'+forcing_name[j]+'*.nc', recursive=True)

                for i, path in enumerate(l):

                    feature_dd[:, :, j, i], _, _, _, _, _ = \
                        read_single_cldas_forcing(path, variable,
                                                  lat_lower, lat_upper,
                                                  lon_left, lon_right)

            feature_dd = np.nanmean(feature_dd, axis=-1)

            # save to nc files
            # ----------------
            f = nc.Dataset(out_path + filename, 'w', format='NETCDF4')

            f.createDimension('longitude', size=Nlon)
            f.createDimension('latitude', size=Nlat)
            f.createDimension('feature', size=len(variables))

            longitude = f.createVariable(
                'longitude', 'f4', dimensions='longitude')
            latitude = f.createVariable(
                'latitude', 'f4', dimensions='latitude')
            force = f.createVariable('forcing', 'f4',
                                     dimensions=('latitude', 'longitude', 'feature'))

            longitude[:] = LON
            latitude[:] = LAT
            force[:] = feature_dd

            f.close()


def preprocess_train_daily_cldas_forcing(input_path, out_path, begin_date, end_date):

    # read CLDAS and preprocess
    CLDAS, lat_, lon_ = read_daily_cldas_forcing(input_path,
                                                 begin_date, end_date)
    CLDAS, min_, max_ = preprocess_daily_data(CLDAS)

    # get dates array according to begin/end dates
    dates = get_date_array(begin_date, end_date)

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
        forcing = f.createVariable('forcing', 'f4',
                                   dimensions=('latitude', 'longitude', 'feature'))
        min = f.createVariable('min', 'f4',
                               dimensions=('latitude', 'longitude', 'feature'))
        max = f.createVariable('max', 'f4',
                               dimensions=('latitude', 'longitude', 'feature'))

        lon[:] = lon_
        lat[:] = lat_
        forcing[:] = CLDAS[i]
        min[:] = min_
        max[:] = max_

        f.close()

# TODO: complete preprocess_test_daily_data in utils.py
def preprocess_test_daily_cldas_forcing(input_path,
                                  input_preprocess_path,
                                  out_path,
                                  begin_date,
                                  end_date):

    # read CLDAS NRT (timestep, lat, lon, feature)
    CLDAS, lat_, lon_ = read_daily_cldas_forcing(
        input_path, begin_date, end_date)

    # get shape
    Nt, Nlat, Nlon, Nf = CLDAS.shape

    # get min/max
    _, _, _, min_, max_ = read_preprocessed_daily_cldas_forcing(
        input_preprocess_path, '2015-03-31', '2015-04-01')

    # preprocess according normalized parameters
    for i in np.arange(Nlat):
        for j in np.arange(Nlon):

            try:
                # interplot
                imp = SimpleImputer(missing_values=np.nan, strategy='mean')
                CLDAS[:, i, j, :] = imp.fit_transform(CLDAS[:, i, j, :])

                # min max scaler
                for m in np.arange(Nf):
                    CLDAS[:, i, j, m] = \
                        (CLDAS[:, i, j, m]-min_[i, j, m]) / \
                        (max_[i, j, m]-min_[i, j, m])
            except:
                print('all data is nan')

    # get dates array according to begin/end dates
    dates = get_date_array(begin_date, end_date)

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
        forcing = f.createVariable('forcing', 'f4',
                                   dimensions=('latitude', 'longitude', 'feature'))
        min = f.createVariable('min', 'f4',
                               dimensions=('latitude', 'longitude', 'feature'))
        max = f.createVariable('max', 'f4',
                               dimensions=('latitude', 'longitude', 'feature'))

        min[:] = min_
        max[:] = max_
        lon[:] = lon_
        lat[:] = lat_
        forcing[:] = CLDAS[i]

        f.close()
