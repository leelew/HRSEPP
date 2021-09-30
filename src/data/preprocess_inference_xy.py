from data.preprocess_train_xy import read_daily_CLDAS_forcing
from data.grids_match_xy import read_p_daily_CLDAS_forcing

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

from data.utils import _get_date_array



def preprocess_CLDAS_NRT(input_path, 
                         input_preprocess_path, 
                         out_path, 
                         begin_date, 
                         end_date):

    # read CLDAS NRT (timestep, lat, lon, feature)
    CLDAS, lat_, lon_ = read_daily_CLDAS_forcing(\
        input_path, begin_date, end_date)

    # get shape
    Nt, Nlat, Nlon, Nf = CLDAS.shape
    
    # get min/max
    _, _, _, min_, max_ = read_p_daily_CLDAS_forcing(\
        input_preprocess_path, '2015-03-31', '2015-04-01')

    # preprocess according normalized parameters
    for i in np.arange(Nlat):
        for j in np.arange(Nlon):

            try:
                # interplot
                imp = SimpleImputer(missing_values=np.nan, strategy='mean')
                CLDAS[:,i,j,:] = imp.fit_transform(CLDAS[:,i,j,:])

                # min max scaler
                for m in np.arange(Nf):
                    CLDAS[:,i,j,m] = \
                        (CLDAS[:,i,j,m]-min_[i,j,m])/(max_[i,j,m]-min_[i,j,m])
            except:
                print('all data is nan')


    # get dates array according to begin/end dates
    dates = _get_date_array(begin_date, end_date)

    # save file (after integrate spatial/temporal dimension)
    # ------------------------------------------------------
    for i, date in enumerate(dates):

        # save to nc files
        filename = 'CLDAS_force_P_{year}{month:02}{day:02}.nc'.\
            format(year=date.year,
                   month=date.month,
                   day=date.day)
        print('now we saving {}'.format(filename))

        f = nc.Dataset(out_path + filename, 'w', format='NETCDF4')

        f.createDimension('longitude', size=lon_.shape[0])
        f.createDimension('latitude', size=lat_.shape[0])
        f.createDimension('feature', size=CLDAS.shape[-1])

        lon = f.createVariable('longitude', 'f4', dimensions='longitude')
        lat = f.createVariable('latitude', 'f4', dimensions='latitude')
        forcing = f.createVariable('forcing', 'f4', \
            dimensions=('latitude', 'longitude', 'feature'))
        min = f.createVariable('min', 'f4', \
            dimensions=('latitude', 'longitude', 'feature'))
        max = f.createVariable('max', 'f4', \
            dimensions=('latitude', 'longitude','feature'))

        min[:] = min_
        max[:] = max_
        lon[:] = lon_
        lat[:] = lat_
        forcing[:] = CLDAS[i]
        
        f.close()