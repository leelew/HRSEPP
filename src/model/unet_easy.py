import os
import sys

import config
import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Input,
                                     MaxPooling2D, UpSampling2D, concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
"""Define U-Net model using IceNet from nature communication
"""


class MaskMSELoss(tf.keras.losses.Loss):
    def __init__(self, mask):
        super().__init__()
        self.mask = mask

    def call(self, y_true, y_pred):
        mse = tf.math.reduce_mean(tf.square(y_true - y_pred), axis=0)
        mask_mse = tf.math.multiply(mse, self.mask)
        return tf.math.reduce_mean(mask_mse)


def unet_easy(input_shape,
              loss,
              mask,
              weighted_metrics,
              learning_rate=1e-4,
              filter_size=3,
              n_filters_factor=1,
              n_forecast_months=7,
              use_temp_scaling=False,
              n_output_classes=1,
              **kwargs):
    inputs = Input(shape=input_shape)

    conv1 = Conv2D(np.int(64 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(np.int(64 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv1)
    bn1 = BatchNormalization(axis=-1)(conv1)  # 112, 112, 64
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)  # 56, 56, 64

    conv2 = Conv2D(np.int(128 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(np.int(128 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv2)
    bn2 = BatchNormalization(axis=-1)(conv2)  # 56, 56, 128
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)  # 28, 28, 128

    conv3 = Conv2D(np.int(256 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(np.int(256 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv3)
    bn3 = BatchNormalization(axis=-1)(conv3)  # 28, 28, 256

    up4 = Conv2D(np.int(128 * n_filters_factor),
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(
                     size=(2, 2), interpolation='nearest')(bn3))  # 56, 56 128
    merge4 = concatenate([bn2, up4], axis=3)
    conv4 = Conv2D(np.int(128 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge4)
    conv4 = Conv2D(np.int(128 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv4)
    bn4 = BatchNormalization(axis=-1)(conv4)

    up5 = Conv2D(np.int(64 * n_filters_factor),
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(
                     size=(2, 2), interpolation='nearest')(bn4))
    merge5 = concatenate([bn1, up5], axis=3)
    conv5 = Conv2D(np.int(64 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge5)
    conv5 = Conv2D(np.int(64 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv5)
    bn5 = BatchNormalization(axis=-1)(conv5)

    final_layer_logits = [(Conv2D(n_output_classes, 1,
                                  activation='linear')(conv5))
                          for i in range(n_forecast_months)]
    final_layer_logits = tf.concat(final_layer_logits, axis=-1)

    model = Model(inputs, final_layer_logits)

    model.compile(optimizer=Adam(lr=learning_rate), loss=MaskMSELoss(mask))
    #weighted_metrics=weighted_metrics)
    model.summary()
    return model


if __name__ == '__main__':
    unet_easy(input_shape=(112, 112, 7), loss='mse', weighted_metrics=0.1)
