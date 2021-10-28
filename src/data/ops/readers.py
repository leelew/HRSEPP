import datetime as dt
import h5py
import numpy as np


class RawSMAPReader():
    """Reader for raw SMAP data.
    
    Returns:
        data: shape as [feature, lat, lon], feature = 1 when reading one variable.
        lat_2d, lon_2d: 2d matrix contain the latitude or longitude for each grid.
        date: datetime of file.
    """
    def __init__(self, lat_lower, lat_upper, lon_left, lon_right):
        self.lat_lower = lat_lower
        self.lat_upper = lat_upper
        self.lon_left = lon_left
        self.lon_right = lon_right

    def read_one_file(self, one_file_path, var_list=['sm_surface']):
        yyyymmddhh = one_file_path.split('/')[-1].split('_')[4]
        yyyy = int(yyyymmddhh[0:4])
        mm = int(yyyymmddhh[4:6])
        dd = int(yyyymmddhh[6:8])
        hh = int(yyyymmddhh[9:11])
        date = dt.datetime(yyyy, mm, dd, hh)
        print('[HRSEPP][IO]Read SMAP of {}'.format(date))

        # handle for HDF file
        f = h5py.File(one_file_path, 'r')

        # read attribute (lat, lon) and crop attributes
        lat, lon = f['cell_lat'][:, 0], f['cell_lon'][0, :]

        lat_idx = np.where((lat > self.lat_lower) & (lat < self.lat_upper))[0]
        lon_idx = np.where((lon > self.lon_left) & (lon < self.lon_right))[0]

        #FIXME:Change this slice method!
        lat_2d = f['cell_lat'][lat_idx, :][:, lon_idx]
        lon_2d = f['cell_lon'][lat_idx, :][:, lon_idx]

        if len(var_list) > 1:
            data = np.zeros((len(var_list), lat_2d.shape[0], lat_2d.shape[1]))
            for i, var in enumerate(var_list):
                data[i] = f['Geophysical_Data'][var][lat_idx, :][:, lon_idx]
        else:
            data = f['Geophysical_Data'][var_list[0]][lat_idx, :][:, lon_idx][
                np.newaxis]

        # turn fillvalue to NaN
        data[data == -9999] = np.nan

        return data, lat_2d, lon_2d


class NCReader():
    pass


if __name__ == '__main__':
    data, lat_2d, lon_2d = RawSMAPReader(
        10, 20, 30, 40
    )('/hard/lilu/SMAP_L4/SMAP_L4/2015.05.31/SMAP_L4_SM_gph_20150531T223000_Vv4030_001.h5'
      )

    print(data.shape)
    print(lat_2d.shape)
    print(lon_2d.shape)

    data, lat_2d, lon_2d = RawSMAPReader(
        10, 20, 30, 40
    )('/hard/lilu/SMAP_L4/SMAP_L4/2015.05.31/SMAP_L4_SM_gph_20150531T223000_Vv4030_001.h5',
      var_list=[
          'precipitation_total_surface_flux',
          'radiation_longwave_absorbed_flux',
          'radiation_shortwave_downward_flux',
          'specific_humidity_lowatmmodlay', 'surface_pressure', 'surface_temp',
          'windspeed_lowatmmodlay'
      ])

    print(data.shape)
    print(lat_2d.shape)
    print(lon_2d.shape)
