# ==============================================================================
# Slices target regions into 1°x1° sub-regions, and construct input/output
# data for different spatial dimension (0.0625 for CLDAS, 9km for SMAP)

# 1. for training mode, we select each data and used neighbor grids of CLDAS
#    as training dataset.
# 2. for inference mode, we select each grid for CLDAS and used neighbor grid
#    of SMAP as inference dataset.
#
# author: Lu Li
# mail: lilu35@mail2.sysu.edu.cn
# ==============================================================================

import netCDF4 as nc
import numpy as np
import glob
import matplotlib.pyplot as plt
from .preprocess import read_p_daily_CLDAS_forcing, read_daily_SMAP


# ------------------------------------------------------------------------------
# 1. make preliminary data for each 1°x1° regions for training mode
# ------------------------------------------------------------------------------
def slices_inputs_task(input_SMAP_path, input_CLDAS_path, out_path,
                       task,
                       begin_date, end_date,
                       lat_lower, lat_upper, 
                       lon_left, lon_right):


    # 选取对应时间预处理后的数据，拼接成(time, lat, lon, Nf)
    # -----------------------------------------------
    CLDAS, CLDAS_lat, CLDAS_lon, _, _ = read_p_daily_CLDAS_forcing(\
        input_CLDAS_path, begin_date, end_date)
    SMAP, SMAP_lat, SMAP_lon = read_daily_SMAP(\
        input_SMAP_path, begin_date, end_date)

    # get shape
    Nt, Nlat, Nlon, Nf = CLDAS.shape

    # 对于一个区域，选取内部所有SMAP L4的点对应的经纬度。
    # ------------------------------------------
    lat_idx = np.where((SMAP_lat>lat_lower) & (SMAP_lat<lat_upper))[0]
    lon_idx = np.where((SMAP_lon>lon_left) & (SMAP_lon<lon_right))[0]

    # 对于选取的每一个点，选取最邻近的CLDAS组成FORCING。
    # ------------------------------------------
    x_train = np.full((len(lat_idx)*len(lon_idx)*Nt,Nf), np.nan) # add SMAP
    y_train = np.full((len(lat_idx)*len(lon_idx)*Nt,1),np.nan)
    count = 0

    for i in lat_idx:
        for j in lon_idx:

            # get selected one grid lat, lon
            SMAP_lat_, SMAP_lon_ = SMAP_lat[i], SMAP_lon[j]

            # get difference of attribute
            d_lat = np.abs(CLDAS_lat-SMAP_lat_) 
            d_lon = np.abs(CLDAS_lon-SMAP_lon_)

            # find corresponding index of nearest grid
            idx_min_lat = np.where(d_lat == np.nanmin(d_lat))[0][0]
            idx_min_lon = np.where(d_lon == np.nanmin(d_lon))[0][0]

            # 将该区域的每个点的FORCING-LABEL合并组成最后的训练集。
            # ---------------------------------------------
            y_train[count*Nt:count*Nt+Nt] = SMAP[:, i, j, :]
            x_train[count*Nt:count*Nt+Nt] = CLDAS[:, idx_min_lat, idx_min_lon,:]
                #np.concatenate(\
                #    [CLDAS[:, idx_min_lat, idx_min_lon,:], SMAP[:, i, j, :]], \
                #        axis=-1)

            count+=1
    


    # 存储该FORCING和LABEL。
    # -------------------
    filename = 'raw_task_{}.nc'.format(task)

    f = nc.Dataset(out_path + filename, 'w', format='NETCDF4')

    f.createDimension('longitude', size=len(lon_idx))
    f.createDimension('latitude', size=len(lat_idx))
    f.createDimension('time', size=x_train.shape[0])
    f.createDimension('feature', size=x_train.shape[1])
    f.createDimension('m', size=1)

    lon = f.createVariable('longitude', 'f4', dimensions='longitude')
    lat = f.createVariable('latitude', 'f4', dimensions='latitude')
    x_train_ = f.createVariable('x_train', 'f4', dimensions=('time', 'feature'))
    y_train_ = f.createVariable('y_train', 'f4', dimensions=('time','m' ))

    lon[:] = SMAP_lon[lon_idx]
    lat[:] = SMAP_lat[lat_idx]
    x_train_[:] = x_train
    y_train_[:] = y_train
    
    f.close()
    

def slices_inputs(input_SMAP_path, input_CLDAS_path, out_path,
                  begin_date, end_date,
                  lat_lower, lat_upper, 
                  lon_left, lon_right, 
                  window_size,):

    task = 0

    # 对于经纬度范围，选择间隔区域，每1°x1°构建一个任务
    for i in np.arange(lat_lower, lat_upper, window_size):
        for j in np.arange(lon_left, lon_right, window_size):
            print(i,j)
            # 对应每个区域，制作该区域的输入数据
            slices_inputs_task(input_SMAP_path,
                            input_CLDAS_path,
                            out_path,
                            task,
                            begin_date, 
                            end_date,
                            i, i+window_size, 
                            j, j+window_size)

            task+=1

# ------------------------------------------------------------------------------
# 2. combine and zip data for LSTM for each 1°x1° regions for training mode
# ------------------------------------------------------------------------------
def make_input(inputs, 
               outputs,
               len_input, 
               len_output, 
               window_size):
    """Generate inputs and outputs for LSTM."""
    # caculate the last time point to generate batch
    end_idx = inputs.shape[0] - len_input - len_output - window_size
    # generate index of batch start point in order
    batch_start_idx = range(end_idx)
    # get batch_size
    batch_size = len(batch_start_idx)
    # generate inputs
    input_batch_idx = [
        (range(i, i + len_input)) for i in batch_start_idx]
    inputs = np.take(inputs, input_batch_idx, axis=0). \
        reshape(batch_size, len_input,
                inputs.shape[1])
    # generate outputs
    output_batch_idx = [
        (range(i + len_input + window_size, i + len_input + window_size +
               len_output)) for i in batch_start_idx]
    outputs = np.take(outputs, output_batch_idx, axis=0). \
        reshape(batch_size,  len_output,
                outputs.shape[1])
    
    return inputs, np.squeeze(outputs)


def make_train_input_task(input_path, out_path, task):
    
    # read train data
    filename = 'raw_task_{}.nc'.format(task)
    f = nc.Dataset(input_path + filename, 'r')
    x_train = f['x_train'][:]
    y_train = f['y_train'][:]

    idx_nonan = np.where(~np.isnan(x_train))[0]
    x_train = x_train[idx_nonan]
    y_train = y_train[idx_nonan]

    idx_nonan = np.where(~np.isnan(y_train))[0]
    x_train = x_train[idx_nonan]
    y_train = y_train[idx_nonan]

    inputs, outputs = make_input(x_train, 
                                 y_train, 
                                 len_input=10, 
                                 len_output=1, 
                                 window_size=3)
    
    print(np.isnan(inputs).any())
    print(np.isnan(outputs).any())

    # get shape
    N, timestep, features = inputs.shape

    # save to nc files
    # ----------------
    filename = 'train_input_task_{}.nc'.format(task)

    f = nc.Dataset(out_path + filename, 'w', format='NETCDF4')

    f.createDimension('sample', size=N)
    f.createDimension('timestep', size=timestep)
    f.createDimension('feature', size=features)

    x_train = f.createVariable('x_train', 'f4', \
        dimensions=('sample','timestep','feature'))
    y_train = f.createVariable('y_train', 'f4', dimensions=('sample', ))

    x_train[:] = inputs
    y_train[:] = outputs


    
    f.close()

def make_train_input(input_path, out_path):

    l = glob.glob(input_path + 'raw_task_*.nc', recursive=True)

    for task in np.arange(len(l)):
        print(task)
        make_train_input_task(input_path, out_path, task)


# ------------------------------------------------------------------------------
# 3. make inference data for each 1°x1° regions for inference mode
# ------------------------------------------------------------------------------
def make_inference_input_task(input_SMAP_path, input_CLDAS_path, out_path,
                              begin_date, end_date,
                              task,
                              lat_lower, lat_upper, 
                              lon_left, lon_right,):

    # 选取对应时间预处理后的数据，拼接成(timestep, lat, lon, Nf)
    # ---------------------------------------------
    CLDAS, CLDAS_lat, CLDAS_lon,_,_ = read_p_daily_CLDAS_forcing(\
        input_CLDAS_path, begin_date, end_date)
    SMAP, SMAP_lat, SMAP_lon = read_daily_SMAP(\
        input_SMAP_path, begin_date, end_date)

    # 对于每个1x1的小区域，选取所有CLDAS的点，并获取对应的经纬度。
    # -------------------------------------------------
    
    # get shape
    Nt, Nlat, Nlon, Nf = CLDAS.shape

    # 对于一个区域，选取内部所有CLDAS的点对应的经纬度。
    lat_idx = np.where((CLDAS_lat>lat_lower) & (CLDAS_lat<lat_upper))[0]
    lon_idx = np.where((CLDAS_lon>lon_left) & (CLDAS_lon<lon_right))[0]

    # 对于选取的每一个点，选取最邻近的SMAP组成FORCING。
    # ------------------------------------------
    x_predict = np.full((len(lat_idx),len(lon_idx),Nt,Nf), np.nan) # add SMAP

    for i, idx_1 in enumerate(lat_idx):
        for  j, idx_2 in enumerate(lon_idx):
            
            # get selected one grid lat, lon
            cldas_lat_, cldas_lon_ = CLDAS_lat[idx_1], CLDAS_lon[idx_2]

            # get difference of attribute
            d_lat = np.abs(SMAP_lat-cldas_lat_), 
            d_lon = np.abs(SMAP_lon-cldas_lon_)

            # find corresponding index of nearest grid
            idx_min_lat = np.where(d_lat == np.nanmin(d_lat))[0][0]
            idx_min_lon = np.where(d_lon == np.nanmin(d_lon))[0][0]

            # 将该区域的每个点的FORCING-LABEL合并组成最后的训练集。
            # ---------------------------------------------

            x_predict[i,j, :,:] = CLDAS[:, idx_1, idx_2,:]
                #np.concatenate(\
                #    [CLDAS[:, idx_1, idx_2,:], \
                #     SMAP[:, idx_min_lat, idx_min_lon, :]], \
                #         axis=-1)

    print(np.isnan(x_predict).any())

    # 存储该FORCING和LABEL。
    # -------------------
    filename = 'predict_task_{}.nc'.format(task)

    f = nc.Dataset(out_path + filename, 'w', format='NETCDF4')

    f.createDimension('longitude', size=len(lon_idx))
    f.createDimension('latitude', size=len(lat_idx))
    f.createDimension('timestep', size=x_predict.shape[-2])
    f.createDimension('feature', size=x_predict.shape[-1])

    lon = f.createVariable('longitude', 'f4', dimensions='longitude')
    lat = f.createVariable('latitude', 'f4', dimensions='latitude')
    x_predict_ = f.createVariable('x_predict', 'f4', \
        dimensions=('latitude','longitude','timestep', 'feature'))

    lon[:] = CLDAS_lon[lon_idx]
    lat[:] = CLDAS_lat[lat_idx]
    x_predict_[:] = x_predict
    
    f.close()


def make_inference_input(input_CLDAS_path, input_SMAP_path, out_path,
                         begin_date, end_date,
                         lat_lower, lat_upper, 
                         lon_left, lon_right, 
                         window_size,):

    task = 0

    # 对于经纬度范围，选择间隔区域，每1°x1°构建一个任务
    for i in np.arange(lat_lower, lat_upper, window_size):
        for j in np.arange(lon_left, lon_right, window_size):

            # 对应每个区域，制作该区域的输入数据
            make_inference_input_task(input_SMAP_path,
                                      input_CLDAS_path,
                                      out_path,
                                      begin_date, 
                                      end_date,
                                      task,
                                      i, i+window_size, 
                                      j, j+window_size,)

            task+=1