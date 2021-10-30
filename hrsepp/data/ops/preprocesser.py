import glob
import json
import os
import sys

from numpy.lib.polynomial import _raise_power

sys.path.append('../../../hrsepp/')
import netCDF4 as nc
import numpy as np
from data.ops.init import AuxManager
from data.ops.readers import RawSMAPReader
from data.ops.saver import nc_saver
from data.ops.time import TimeManager
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


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
                     X=np.nanmean(tmp, axis=0),
                     var_name=self.var_name,
                     date=date,
                     lat_2d=self.aux['lat_2d'],
                     lon_2d=self.aux['lon_2d'])

        return data


class Preprocesser():
    def __init__(self,
                 X,
                 save_path,
                 auxiliary_path,
                 begin_date,
                 end_date,
                 mode='train',
                 save=True,
                 var_name='SSM'):

        # get shape
        self.Nt, self.Nf, self.Nlat, self.Nlon = X.shape
        self.begin_date = begin_date
        self.end_date = end_date
        self.auxiliary_path = auxiliary_path
        with open(auxiliary_path + 'auxiliary.json', 'r') as f:
            self.aux = json.load(f)
        self.save_path = save_path
        self.mode = mode
        self.save = save
        self.X = X
        self.var_name = var_name

    def __call__(self):
        if self.mode == 'train':
            X, min_scale, max_scale = self._train_preprocesser(self.X)
            AuxManager().update(self.auxiliary_path, 'min_scale',
                                min_scale.tolist())
            AuxManager().update(self.auxiliary_path, 'max_scale',
                                max_scale.tolist())

        else:
            X = self._test_preprocesser(self.X)

        if self.save:
            # get dates array according to begin/end dates
            dates = TimeManager().get_date_array(self.begin_date,
                                                 self.end_date)

            for i, date in enumerate(dates):
                nc_saver(self.save_path, 'p_' + self.var_name, date,
                         self.aux['lon_2d'], self.aux['lat_2d'], X[i])

        return X

    def _train_preprocesser(self, inputs):

        # interplot and scale for each feature on each grid
        min_scale = np.full((self.Nf, self.Nlat, self.Nlon), np.nan)
        max_scale = np.full((self.Nf, self.Nlat, self.Nlon), np.nan)

        # interplot on time dimension.
        for i in range(self.Nlat):
            for j in range(self.Nlon):

                try:
                    # interplot
                    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
                    inputs[:, :, i, j] = imp.fit_transform(inputs[:, :, i, j])

                    # min max scaler
                    scaler = MinMaxScaler()
                    inputs[:, :, i, j] = scaler.fit_transform(inputs[:, :, i,
                                                                     j])
                    min_scale[:, i, j] = scaler.data_min_
                    max_scale[:, i, j] = scaler.data_max_
                except:  # all missing data along time dimension
                    pass

        # interplot on spatial dimension, in order to fill gaps of images.
        for m in range(self.Nt):
            for n in range(self.Nf):

                # interplot
                tmp = inputs[m, n, :, :]
                tmp[np.isnan(tmp)] = np.nanmean(tmp)
                inputs[m, n, :, :] = tmp

        return inputs, min_scale, max_scale

    def _test_preprocesser(self, inputs):

        try:
            min_scale = self.aux['min_scale']
            max_scale = self.aux['max_scale']

            # preprocess according normalized parameters
            for i in range(self.Nlat):
                for j in range(self.Nlon):

                    # interplot
                    imp = SimpleImputer(missing_values=np.nan, strategy='mean')

                    inputs[:, :, i, j] = imp.fit_transform(inputs[:, :, i, j])

                    # min max scaler
                    for m in np.arange(self.Nf):
                        inputs[:, m, i, j] = \
                            (inputs[:, m, i, j]-min_scale[m, i, j]) / \
                            (max_scale[m, i, j]-min_scale[m, i, j])

        except:
            raise IOError('preprocess train data before preprocess test data!')

        return inputs


if __name__ == '__main__':

    AuxManager().init(raw_data_path='/hard/lilu/SMAP_L4/SMAP_L4/',
                      auxiliary_data_path='/hard/lilu/SMAP_L4/test/',
                      lat_lower=10,
                      lat_upper=20,
                      lon_left=30,
                      lon_right=40)

    with open('/hard/lilu/SMAP_L4/test/auxiliary.json') as f:
        aux = json.load(f)

    print(aux.keys())

    data = RawSMAPPreprocesser(raw_data_path='/hard/lilu/SMAP_L4/SMAP_L4/',
                               aux=aux,
                               save_path='/hard/lilu/SMAP_L4/test/SSM/',
                               begin_date='2015-05-31',
                               end_date='2015-06-31',
                               var_name='SSM',
                               var_list=['sm_surface'],
                               lat_lower=10,
                               lat_upper=20,
                               lon_left=30,
                               lon_right=40,
                               save=True)()

    print(data.shape)

    data = RawSMAPPreprocesser(raw_data_path='/hard/lilu/SMAP_L4/SMAP_L4/',
                               aux=aux,
                               save_path='/hard/lilu/SMAP_L4/test/forcing/',
                               begin_date='2015-05-31',
                               end_date='2015-06-31',
                               var_name='forcing',
                               var_list=[
                                   'precipitation_total_surface_flux',
                                   'radiation_longwave_absorbed_flux',
                                   'radiation_shortwave_downward_flux',
                                   'specific_humidity_lowatmmodlay',
                                   'surface_pressure', 'surface_temp',
                                   'windspeed_lowatmmodlay'
                               ],
                               lat_lower=10,
                               lat_upper=20,
                               lon_left=30,
                               lon_right=40,
                               save=True)()

    print(data.shape)

    data = Preprocesser(data,
                        save_path='/hard/lilu/SMAP_L4/test/preprocess/',
                        auxiliary_path='/hard/lilu/SMAP_L4/test/',
                        begin_date='2015-05-31',
                        end_date='2015-06-31',
                        mode='train',
                        save=True,
                        var_name='forcing')()
    print(np.isnan(data).any())
