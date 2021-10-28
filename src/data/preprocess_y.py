import glob
import os

import netCDF4 as nc
import numpy as np

from data.read_smap import read_single_smap
from data.utils import get_date_array


def preprocess_raw_smap(input_path,
                        out_path,
                        begin_date,
                        end_date,
                        lat_lower=-90,
                        lat_upper=90,
                        lon_left=-180,
                        lon_right=180):
    """Read multiple files of SMAP according begin/end dates and integrate time

    Args:
        input_path ([type]): [description]
        out_path ([type]): [description]
        begin_date ([type]): [description]
        end_date ([type]): [description]
        lat_lower (int, optional): [description]. Defaults to -90.
        lat_upper (int, optional): [description]. Defaults to 90.
        lon_left (int, optional): [description]. Defaults to -180.
        lon_right (int, optional): [description]. Defaults to 180.
    """

    # get dates array according to begin/end dates
    dates = get_date_array(begin_date, end_date)

    # read and save file (after integrate spatial/temporal dimension)
    for date in dates:

        # folder name
        foldername = '{year}.{month:02}.{day:02}/'.format(year=date.year,
                                                          month=date.month,
                                                          day=date.day)

        # file list in each folder
        l = glob.glob(input_path + foldername + 'SMAP_L4*.h5', recursive=True)

        # save to nc files
        filename = 'SMAP_L4_SSM_{year}{month:02}{day:02}.nc'.format(
            year=date.year, month=date.month, day=date.day)

        # judge already exist file
        if os.path.exists(out_path + filename):
            print('SMAP SSM of {} already exist'.format(date))
        else:

            # get shape
            _, lat_2d, lon_2d = read_single_smap(l[0], lat_lower, lat_upper,
                                                 lon_left, lon_right)

            # integrate from 3-hour to daily
            ssm_3hh = np.full((lat_2d.shape[0], lon_2d.shape[1], len(l)),
                              np.nan)

            for i, path in enumerate(l):
                ssm_3hh[:, :,
                        i], _, _ = read_single_smap(path, lat_lower, lat_upper,
                                                    lon_left, lon_right)

            ssm_dd = np.nanmean(ssm_3hh, axis=-1)

            # save to nc
            f = nc.Dataset(out_path + filename, 'w', format='NETCDF4')

            f.createDimension('longitude', size=lon_2d.shape[1])
            f.createDimension('latitude', size=lat_2d.shape[0])

            lon = f.createVariable('longitude', 'f4', dimensions='longitude')
            lat = f.createVariable('latitude', 'f4', dimensions='latitude')
            ssm = f.createVariable('ssm',
                                   'f4',
                                   dimensions=('latitude', 'longitude'))

            lon[:], lat[:], ssm[:] = lon_2d[0, :], lat_2d[:, 0], ssm_dd

            f.close()
