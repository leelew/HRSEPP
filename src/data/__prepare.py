# ==============================================================================
# Read SMAP, CLDAS and other data. (provide for training and inference mode)
#   and crop spatial dimension and integrate temporal dimension.
#
# author: Lu Li
# mail: lilu35@mail2.sysu.edu.cn
# ==============================================================================

import datetime as dt
import glob

import h5py
import netCDF4 as nc
import numpy as np


# ------------------------------------------------------------------------------
# 1. read single file of SMAP and crop spatial dimension
# ------------------------------------------------------------------------------
def read_single_SMAP(path, 
                     lat_lower, 
                     lat_upper, 
                     lon_left, 
                     lon_right):

    # get date for data
    yyyymmddhh = path.split('/')[-1].split('_')[4] 
    yyyy = int(yyyymmddhh[0:4])
    mm = int(yyyymmddhh[4:6])
    dd = int(yyyymmddhh[6:8])
    hh = int(yyyymmddhh[9:11])
    date = dt.datetime(yyyy, mm, dd, hh)
    print('Now reading SMAP Level 4 Surface Soil Moisture for {}'.format(date))
    
    # handle for HDF file
    f = h5py.File(path, 'r')

    # read lat, lon
    lat_matrix_global = f['cell_lat'][:]
    lon_matrix_global = f['cell_lon'][:]

    lat = lat_matrix_global[:,0]
    lon = lon_matrix_global[0,:]

    # crop regions according lat/lon
    lat_idx = np.where((lat > lat_lower) & (lat < lat_upper))[0]
    lon_idx = np.where((lon > lon_left) & (lon < lon_right))[0]

    # read surface soil moisture
    ssm = f['Geophysical_Data']['sm_surface'][lat_idx,:][:,lon_idx] 
    ssm[ssm==-9999] = np.nan

    lat_matrix = lat_matrix_global[lat_idx][:,lon_idx]
    lon_matrix = lon_matrix_global[lat_idx][:,lon_idx]

    return ssm, lat_matrix, lon_matrix, \
           date, \
           lat_matrix.shape[0], lat_matrix.shape[1]


# ------------------------------------------------------------------------------
# 2. read multiple files of SMAP according begin/end dates and integrate time
# ------------------------------------------------------------------------------

# get array of date according to begin/end date used for select data range.
def _get_date_array(begin_date, end_date):

    # Initialize the list from begin_date to end_date
    dates = []
    # Initialize the timeindex for append in dates array.
    _dates = dt.datetime.strptime(begin_date, "%Y-%m-%d")
    # initialized the timeindex for decide whether break loop
    _date = begin_date[:]
    # main loop
    while _date <= end_date:
        # pass date in the array
        dates.append(_dates)
        # refresh date by step 1
        _dates = _dates + dt.timedelta(1)
        # changed condition by step 1
        _date = _dates.strftime("%Y-%m-%d")

    return dates

def prepare_SMAP(input_path,
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
    for date in dates:
        
        # folder name
        foldername = '{year}.{month:02}.{day:02}/'.\
            format(year=date.year,
                   month=date.month,
                   day=date.day)
            
        # file list in each folder
        l = glob.glob(input_path + foldername + 'SMAP_L4*.h5', recursive=True)

        # get shape
        _,_,_,_,Nlat,Nlon = read_single_SMAP(l[0], 
                                             lat_lower, lat_upper, 
                                             lon_left, lon_right)
        
        # integrate from 3-hour to daily
        ssm_dd = np.full((Nlat, Nlon, len(l)), np.nan)

        for i, path in enumerate(l):
            ssm_dd[:,:,i], lat_matrix, lon_matrix,_,_,_ = \
                read_single_SMAP(path, 
                                 lat_lower, lat_upper, 
                                 lon_left, lon_right)

        ssm_dd = np.nanmean(ssm_dd, axis=-1)

        # save to nc files
        filename = 'SMAP_L4_SSM_{year}{month:02}{day:02}.nc'.\
            format(year=date.year,
                   month=date.month,
                   day=date.day)

        f = nc.Dataset(out_path + filename, 'w', format='NETCDF4')

        f.createDimension('longitude', size=Nlon)
        f.createDimension('latitude', size=Nlat)

        lon = f.createVariable('longitude', 'f4', dimensions='longitude')
        lat = f.createVariable('latitude', 'f4', dimensions='latitude')
        ssm = f.createVariable('ssm', 'f4', dimensions=('latitude', 'longitude'))

        lon[:] = lon_matrix[0,:]
        lat[:] = lat_matrix[:,0]
        ssm[:] = ssm_dd
        
        f.close()


# ------------------------------------------------------------------------------
# 3. Read CLDAS forcing and crop spatial dimension
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
# 4. Read multiple CLDAS forcing and integrate temporal dimension
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
        filename = 'CLDAS_force_{year}{month:02}{day:02}.nc'.\
            format(year=date.year,
                month=date.month,
                day=date.day)

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
# 5. Read multiple CLDAS model and integrate temporal dimension
# ------------------------------------------------------------------------------
def prepare_CLDAS_model(input_path,
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
        filename = 'CLDAS_force_{year}{month:02}{day:02}.nc'.\
            format(year=date.year,
                month=date.month,
                day=date.day)

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