import sys

sys.path.append('../../hrsepp/')

import numpy as np
import tensorflow as tf
import wandb
from sklearn.metrics import r2_score
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from ..model.convlstm import ed_convlstm, convlstm1, convlstm3
from ..model.unet import unet5, unet9
from ..utils.callback import CallBacks
from data.data_loader import DataLoader
import json


def train(x_train,
          y_train,
          x_valid,
          y_valid,
          mask,
          model_name,
          save_path,
          wanda_mode=False):

    wandb.init()

    # wandb setting
    default = dict(learning_rate=0.008132,
                   n_filters_factor=1.111,
                   filter_size=3,
                   batch_size=2)

    wandb.init(config=default)

    # model
    model = ed_convlstm(input_shape=(224, 224, 8),
                        mask=mask,
                        learning_rate=wandb.config.learning_rate,
                        filter_size=wandb.config.filter_size,
                        n_filters_factor=wandb.config.n_filters_factor,
                        n_forecast_months=7,
                        n_output_classes=1)

    model.fit(x_train,
              y_train,
              batch_size=wandb.config.batch_size,
              epochs=wandb.config.epochs,
              callbacks=CallBacks(),
              validation_split=0.2)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_valid)

    np.save('y_train_pred', y_train_pred)
    np.save('y_test_pred', y_test_pred)

    model.save(save_path)


if __name__ == '__main__':

    x_train = np.load('/hard/lilu/x_train_1.npy')
    y_train = np.load('/hard/lilu/y_train_1.npy')

    x_test = np.load('/hard/lilu/x_test_1.npy')
    y_test = np.load('/hard/lilu/y_test_1.npy')

    x_tr, y_tr = DataLoader(len_input=1,
                            len_output=7,
                            window_size=1,
                            use_lay_y=True,
                            mode='train')(x_train, y_train)

    x_te, y_te = DataLoader(len_input=1,
                            len_output=7,
                            window_size=1,
                            use_lay_y=True,
                            mode='train')(x_test, y_test)

    with open('/hard/lilu/SMAP_L4/auxiliary.json') as f:
        aux = json.load(f)

    train(x_tr,
          y_tr,
          x_te,
          y_te,
          np.array(aux['mask']),
          model_name='ed_convlstm',
          save_path='/hard/lilu/SMAP_L4/model/ed_convlstm')
