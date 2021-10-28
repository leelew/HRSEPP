from src.data.ops.time import TimeManager
import glob
import netCDF4 as nc
import os
import json
import numpy as np
from src.data.ops.readers import RawSMAPReader
from src.data.ops.saver import nc_saver


class RawSMAPPreprocesser():
    def __init__(self,
                 raw_data_path,
                 aux,
                 out_path,
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
        self.out_path = out_path
        self.begin_date = begin_date
        self.end_date = end_date
        self.save = save
        self.var_name = var_name
        self.var_list = var_list
        self.save_path = save_path

        self.raw_smap_reader = RawSMAPReader(lat_lower, lat_upper, lon_left,
                                             lon_right)()

    def preprocess_raw_data(self):
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

            # integrate from 3-hour to daily #NOTE:Only suite for SMAP
            tmp = np.full((self.aux['Nlat'], self.aux['Nlon'], len(l)), np.nan)

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
