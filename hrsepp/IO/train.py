import numpy as np
import tensorflow as tf
import wandb
from sklearn.metrics import r2_score
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from ..model.convlstm import convlstm1, convlstm3
from ..model.unet import unet5, unet9
from ..utils.callback import CallBacks


def train(x_train,
          y_train,
          x_valid,
          y_valid,
          mask,
          model_name,
          save_path,
          wanda_mode=False):

    if wanda_mode:
        wandb.init()

    model = unet5(input_shape=(112, 112, 8),
                  mask=mask,
                  learning_rate=wandb.config.learning_rate,
                  filter_size=wandb.config.filter_size,
                  n_filters_factor=wandb.config.n_filters_factor,
                  n_forecast_months=1,
                  n_output_classes=1)

    model.fit(x_train,
              y_train,
              batch_size=wandb.config.batch_size,
              epochs=wandb.config.epochs,
              callbacks=CallBacks(),
              validation_data=[x_valid, y_valid])

    model.save(save_path)
