# ==============================================================================
# train DL models
#
# author: Lu Li
# mail: lilu35@mail2.sysu.edu.cn
# ==============================================================================
# TODO: split train/test/inference data, give a parameter
# to control restart train or test or inference

import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score

from data.make_inference_xy import make_inference_data
from data.make_test_xy import make_test_data
from data.make_train_xy import make_train_data
#from IO.train import keras_train, load_model
from model.lstm import lstm
from model.convlstm import convlstm
from utils.config import parse_args

"""GPU setting module
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')

for gpu in gpus:
    tf.config.experimental.set_visible_devices(
        devices=gpus[1], device_type='GPU')
    tf.config.experimental.set_memory_growth(
        device=gpu, enable=True)
"""


def main(mode):

    config = parse_args()

    if mode == 'train':
        # ----------------------------------------------------------------------
        # 1. train mode (re-train once a month)
        # ----------------------------------------------------------------------
        X, y = make_train_data(
            raw_X_path=os.path.join(
                config.ROOT, config.x_path, config.raw_x_path),
            raw_y_path=os.path.join(
                config.ROOT, config.y_path, config.raw_y_path),
            daily_X_path=os.path.join(
                config.ROOT, config.x_path, config.daily_x_path),
            daily_y_path=os.path.join(
                config.ROOT, config.y_path, config.daily_y_path),
            begin_date=config.begin_train_date,
            end_date=config.end_train_date,
            lat_lower=config.lat_lower,
            lat_upper=config.lat_upper,
            lon_left=config.lon_left,
            lon_right=config.lon_right,

            len_input=config.len_input,
            len_output=config.len_output,
            window_size=config.window_size,
            use_lag_y=config.use_lag_y)

        import numpy as np
        np.save('x_train.npy', X)
        np.save('y_train.npy', y)
        #model = lstm(
        #    n_feature=X.shape[-1],
        #    input_len=config.len_input)
        # model = convlstm(
        #    n_feature=X.shape[-1],
        #    input_len=config.len_input,
        #    n_lat=X.shape[-2],
        #    n_lon=X.shape[-3])

        #keras_train(model,
        #            X, y,
        #            batch_size=config.batch_size,
        #            epochs=config.epoch,
        #            save_folder=os.path.join(config.ROOT, config.saved_model_path))

    elif mode == 'test':
        # ----------------------------------------------------------------------
        # 2. test mode (optional)
        # ----------------------------------------------------------------------
        X, y = make_test_data(
            raw_X_path=os.path.join(
                config.ROOT, config.x_path, config.raw_x_path),
            raw_y_path=os.path.join(
                config.ROOT, config.y_path, config.raw_y_path),
            daily_X_path=os.path.join(
                config.ROOT, config.x_path, config.daily_x_path),
            daily_y_path=os.path.join(
                config.ROOT, config.y_path, config.daily_y_path),
            begin_date=config.begin_test_date,
            end_date=config.end_test_date,
            lat_lower=config.lat_lower,
            lat_upper=config.lat_upper,
            lon_left=config.lon_left,
            lon_right=config.lon_right,

            len_input=config.len_input,
            len_output=config.len_output,
            window_size=config.window_size,
            use_lag_y=config.use_lag_y)

        import numpy as np
        np.save('x_test.npy', X)
        np.save('y_test.npy', y)
        #model = load_model(os.path.join(config.ROOT, config.saved_model_path))

        #for i in range(X.shape[2]):
        #    for j in range(X.shape[3]):

        #        y_pred = model.predict(X[:, :, i, j, :])
        #        print(r2_score(np.squeeze(y[:, :, i, j, :]), y_pred))

    elif mode == 'inference':
        # ----------------------------------------------------------------------
        # 3. inference mode (once a day)
        # ----------------------------------------------------------------------
        X = make_inference_data(
            raw_X_path=os.path.join(
                config.ROOT, config.x_path, config.raw_x_path),
            raw_y_path=os.path.join(
                config.ROOT, config.y_path, config.raw_y_path),
            daily_X_path=os.path.join(
                config.ROOT, config.x_path, config.daily_x_path),
            daily_y_path=os.path.join(
                config.ROOT, config.y_path, config.daily_y_path),
            begin_date=config.begin_inference_date,
            end_date=config.end_inference_date,
            lat_lower=config.lat_lower,
            lat_upper=config.lat_upper,
            lon_left=config.lon_left,
            lon_right=config.lon_right,

            len_input=config.len_input,
            len_output=config.len_output,
            window_size=config.window_size,
            use_lag_y=config.use_lag_y)

        print(X.shape)


if __name__ == '__main__':
    main(mode='test')
