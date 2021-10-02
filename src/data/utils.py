import datetime as dt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

from data.read_cldas import read_preprocessed_daily_cldas_forcing


def get_date_array(begin_date, end_date):
    """get array of date according to begin/end date used for select data range.


    Args:
        begin_date ([type]): [description]
        end_date ([type]): [description]

    Returns:
        [type]: [description]
    """

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


# ------------------------------------------------------------------------------
# 2. Preprocessing data by interplot and normalization for training mode
#    Note: preprocessing process isn't applied on SMAP.
# ------------------------------------------------------------------------------
def preprocess_train_daily_data(inputs):

    # get shape
    Nt, Nlat, Nlon, Nf = inputs.shape

    # interplot and scale for each feature on each grid
    # -------------------------------------------------
    inputs_min = np.full((Nlat, Nlon, Nf), np.nan)
    inputs_max = np.full((Nlat, Nlon, Nf), np.nan)

    for i in np.arange(Nlat):
        for j in np.arange(Nlon):

            try:
                # interplot
                imp = SimpleImputer(missing_values=np.nan, strategy='mean')
                inputs[:, i, j, :] = imp.fit_transform(inputs[:, i, j, :])

                # min max scaler
                scaler = MinMaxScaler()
                inputs[:, i, j, :] = scaler.fit_transform(inputs[:, i, j, :])
                inputs_min[i, j, :] = scaler.data_min_
                inputs_max[i, j, :] = scaler.data_max_
            except:
                print('all data is nan')

    return inputs, inputs_min, inputs_max


def preprocess_test_daily_data(inputs, input_preprocess_path):

    # get shape
    Nt, Nlat, Nlon, Nf = inputs.shape

    # get min/max
    _, _, _, min_, max_ = read_preprocessed_daily_cldas_forcing(
        input_preprocess_path, '2015-03-31', '2015-04-01')

    # preprocess according normalized parameters
    for i in np.arange(Nlat):
        for j in np.arange(Nlon):

            try:
                # interplot
                imp = SimpleImputer(missing_values=np.nan, strategy='mean')
                inputs[:, i, j, :] = imp.fit_transform(inputs[:, i, j, :])

                # min max scaler
                for m in np.arange(Nf):
                    inputs[:, i, j, m] = \
                        (inputs[:, i, j, m]-min_[i, j, m]) / \
                        (max_[i, j, m]-min_[i, j, m])
            except:
                print('all data is nan')

    return inputs
