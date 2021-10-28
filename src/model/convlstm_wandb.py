import os
import sys

import config
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.layers.core import Dense
import xarray as xr
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Input,
                                     ConvLSTM2D, MaxPooling2D, UpSampling2D,
                                     Dense, concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from model.unet import MaskMSELoss


def convlstm_wandb(input_shape,
                   loss,
                   mask,
                   weighted_metrics=0.1,
                   learning_rate=1e-4,
                   filter_size=3,
                   n_filters_factor=1,
                   n_forecast_months=7,
                   use_temp_scaling=False,
                   n_output_classes=1,
                   **kwargs):
    inputs = Input(shape=input_shape)

    c1 = ConvLSTM2D(np.int(16 * n_filters_factor),
                    filter_size,
                    return_sequences=True,
                    activation='relu',
                    padding='same',
                    kernel_initializer='he_normal')(inputs)
    bn1 = BatchNormalization(axis=-1)(c1)

    c2 = ConvLSTM2D(np.int(32 * n_filters_factor),
                    filter_size,
                    return_sequences=True,
                    activation='relu',
                    padding='same',
                    kernel_initializer='he_normal')(bn1)
    bn2 = BatchNormalization(axis=-1)(c2)

    c3 = ConvLSTM2D(np.int(64 * n_filters_factor),
                    filter_size,
                    return_sequences=False,
                    activation='relu',
                    padding='same',
                    kernel_initializer='he_normal')(bn2)
    bn3 = BatchNormalization(axis=-1)(c3)

    out = Dense(1, activation=None)(bn3)

    model = Model(inputs, out)

    model.compile(optimizer=Adam(lr=learning_rate), loss=MaskMSELoss(mask))
    model.summary()
    return model


if __name__ == '__main__':
    convlstm_wandb(input_shape=(1, 112, 112, 8), loss='mse')
