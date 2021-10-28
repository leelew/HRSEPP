from numpy.core.numeric import full
from readers import RawSMAPReader
import json
import glob
import os
import numpy as np


class Init():
    def __init__(self, raw_data_path, auxiliary_data_path, lat_lower,
                 lat_upper, lon_left, lon_right):

        l = glob.glob(raw_data_path + '/2015.05.31/' + 'SMAP*h5',
                      recursive=True)
        data, lat, lon = RawSMAPReader(lat_lower, lat_upper, lon_left,
                                       lon_right)(l[0])
        mask = self.get_mask(data)

        # init attribute and land mask
        attr = {
            'Nlat': lat.shape[0],
            'Nlon': lon.shape[1],
            'mask': mask,
            'lat_2d': lat,
            'lon_2d': lon
        }

        full_path = os.path.join(auxiliary_data_path, 'auxiliary.json')
        with open(full_path, 'w') as f:
            json_string = json.dumps(attr)
            f.write(json_string)

    @staticmethod
    def get_mask(data):
        mask = np.ones_like(data)
        mask[np.isnan(data)] = 0
        return np.array(mask)


if __name__ == '__main__':
    Init(raw_data_path='/hard/lilu/SMAP_L4/SMAP_L4/',
         auxiliary_data_path='/work/lilu/HRSEPP/test/',
         lat_lower=10,
         lat_upper=20,
         lon_left=30,
         lon_right=40)
