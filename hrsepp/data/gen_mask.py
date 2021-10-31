import numpy as np
import netCDF4 as nc


def gen_mask(
        # after reading and process path
        file_path):

    f = nc.Dataset(file_path)
    mask = np.ones_like(f['ssm'][:])
    mask[np.isnan(f['ssm'][:])] = 0
    np.save('mask.npy', np.array(mask))
    return mask


if __name__ == '__main__':
    gen_mask('/hard/lilu/SMAP_L4/test/SSM/SMAP_L4_SSM_20151126.nc')
