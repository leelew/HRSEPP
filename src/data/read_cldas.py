# ==============================================================================
# Read CLDAS and crop spatial dimension and integrate temporal dimension.
#
# author: Lu Li, 2021/09/27
# ============================================================================== 

import datetime as dt
import glob
import os

import h5py
import netCDF4 as nc
import numpy as np

from data.utils import _get_date_array


# ------------------------------------------------------------------------------
# 1. Read CLDAS forcing and crop spatial dimension
# ------------------------------------------------------------------------------
def read_single_CLDAS(path,
                      variable,
                      lat_lower, 
                      lat_upper, 
                      lon_left, 
                      lon_right):
    # get date for data
    yyyymmddhh = path.split('-')[-1].split('.')[0]
    yyyy = int(yyyymmddhh[0:4])
    mm = int(yyyymmddhh[4:6])
    dd = int(yyyymmddhh[6:8])
    hh = int(yyyymmddhh[9:11])
    date = dt.datetime(yyyy, mm, dd, hh)
    print('Now reading CLDAS for {}'.format(date))
    
    # handle for HDF file
    f = h5py.File(path, 'r')

    # read lat, lon
    lat = f['LAT'][:]
    lon = f['LON'][:]

    # crop regions according lat/lon
    lat_idx = np.where((lat > lat_lower) & (lat < lat_upper))[0]
    lon_idx = np.where((lon > lon_left) & (lon < lon_right))[0]

    # read surface soil moisture
    
    data = f[variable][lat_idx,:][:,lon_idx] 
    data[data<-99] = np.nan

    lat_region = lat[lat_idx]
    lon_region = lon[lon_idx]

    return data, lat_region, lon_region, date, \
        lat_region.shape[0], lon_region.shape[0]


# ------------------------------------------------------------------------------
# 2. Read multiple CLDAS forcing and integrate temporal dimension
# ------------------------------------------------------------------------------
def prepare_CLDAS_forcing(input_path,
                          out_path,
                          begin_date, 
                          end_date, 
                          lat_lower=-90, 
                          lat_upper=90, 
                          lon_left=-180, 
                          lon_right=180):
    
    # get dates array according to begin/end dates
    dates = _get_date_array(begin_date, end_date)

    # read and save file (after integrate spatial/temporal dimension)
    # ---------------------------------------------------------------
    variables = ['PRCP','PAIR','QAIR','SWDN','TAIR','WIND']
    forcing_name = ['PRE','PRS','SHU','SSRA','TMP','WIN']

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
            _,LAT,LON,_,Nlat,Nlon = read_single_CLDAS(l[0],
                                                'PRCP', 
                                                    lat_lower, lat_upper, 
                                                    lon_left, lon_right)   
    
            # integrate from 3-hour to daily
            feature_dd = np.full((Nlat, Nlon,len(variables),len(l)), np.nan)

            for j, variable in enumerate(variables):

                # file list in each folder
                l = glob.glob(input_path + \
                    foldername + '*'+forcing_name[j]+'*.nc', recursive=True)

                for i, path in enumerate(l):

                    feature_dd[:,:,j,i],_,_,_,_,_ = \
                        read_single_CLDAS(path, variable, 
                                        lat_lower, lat_upper, 
                                        lon_left, lon_right)

            feature_dd = np.nanmean(feature_dd, axis=-1)
	
            # save to nc files
            # ----------------
            f = nc.Dataset(out_path + filename, 'w', format='NETCDF4')

            f.createDimension('longitude', size=Nlon)
            f.createDimension('latitude', size=Nlat)
            f.createDimension('feature', size=len(variables))

            longitude = f.createVariable('longitude', 'f4', dimensions='longitude')
            latitude = f.createVariable('latitude', 'f4', dimensions='latitude')
            force = f.createVariable('forcing', 'f4', \
                dimensions=('latitude', 'longitude', 'feature'))

            longitude[:] = LON
            latitude[:] = LAT
            force[:] = feature_dd
            
            f.close()


# ------------------------------------------------------------------------------
# 3. Read multiple CLDAS model and integrate temporal dimension
# ------------------------------------------------------------------------------
def prepare_CLDAS_model(input_path,
                        out_path,
                        begin_date, 
                        end_date, 
                        lat_lower=-90, 
                        lat_upper=90, 
                        lon_left=-180, 
                        lon_right=180):
    
    pass