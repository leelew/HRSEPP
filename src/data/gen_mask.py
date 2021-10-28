import numpy as np
import netCDF4 as nc


def gen_mask(
        # after reading and process path
        file_path):

    f = nc.Dataset(file_path)
    mask = np.isnan(f['ssm'])
    mask[mask == False] = 1
    mask[mask == True] = 0

    return mask
