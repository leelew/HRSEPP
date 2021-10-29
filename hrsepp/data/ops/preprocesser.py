import glob
import json
import os
import sys

sys.path.append('../../../hrsepp/')
import netCDF4 as nc
import numpy as np
from data.ops.readers import RawSMAPReader
from data.ops.saver import nc_saver
from data.ops.time import TimeManager
from data.ops.init import Init


class RawSMAPPreprocesser():
    def __init__(self,
                 raw_data_path,
                 aux,
                 save_path,
                 begin_date,
                 end_date,
                 var_name,
                 var_list,
                 lat_lower=-90,
                 lat_upper=90,
                 lon_left=-180,
                 lon_right=180,
                 save=True) -> None:

        self.raw_data_path = raw_data_path
        self.aux = aux
        self.begin_date = begin_date
        self.end_date = end_date
        self.save = save
        self.var_name = var_name
        self.var_list = var_list
        self.save_path = save_path

        self.raw_smap_reader = RawSMAPReader(lat_lower, lat_upper, lon_left,
                                             lon_right)

    def __call__(self):
        # get dates array according to begin/end dates
        dates = TimeManager().get_date_array(self.begin_date, self.end_date)

        # read and save file (after integrate spatial/temporal dimension)
        data = np.full((len(dates), len(
            self.var_list), self.aux['Nlat'], self.aux['Nlon']), np.nan)

        for t, date in enumerate(dates):

            # folder name
            foldername = '{year}.{month:02}.{day:02}/'.format(year=date.year,
                                                              month=date.month,
                                                              day=date.day)
            # file list in each folder
            l = glob.glob(self.raw_data_path + foldername + 'SMAP*.h5',
                          recursive=True)

            assert len(l) == 8, '[HRSEPP][error]lack data of {}'.format(date)

            # integrate from 3-hour to daily #NOTE:Only suite for SMAP
            tmp = np.full((len(l), len(
                self.var_list), self.aux['Nlat'], self.aux['Nlon']), np.nan)

            for i, one_file_path in enumerate(l):

                tmp[i, :, :, :], _, _ = self.raw_smap_reader(
                    one_file_path, self.var_list)

            data[t] = np.nanmean(tmp, axis=0)

            nc_saver(save_path=self.save_path,
                     data=np.nanmean(tmp, axis=0),
                     var_name=self.var_name,
                     date=date,
                     lat=self.aux['lat_2d'],
                     lon=self.aux['lon_2d'])

        return data


if __name__ == '__main__':

    with open('/hard/lilu/SMAP_L4/test/auxiliary.json') as f:
        aux = json.load(f)

    data = RawSMAPPreprocesser(raw_data_path='/hard/lilu/SMAP_L4/SMAP_L4/',
                               aux=aux,
                               save_path='/hard/lilu/SMAP_L4/test/SSM/',
                               begin_date='2015-05-31',
                               end_date='2015-06-01',
                               var_name='SSM',
                               var_list=['sm_surface'],
                               lat_lower=10,
                               lat_upper=20,
                               lon_left=30,
                               lon_right=40,
                               save=True)()

    print(data.shape)
