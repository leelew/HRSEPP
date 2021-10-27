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


def unet_batchnorm(input_shape,
                   loss,
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
    bn1 = BatchNormalization(axis=-1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)

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
    bn2 = BatchNormalization(axis=-1)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)

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
    bn3 = BatchNormalization(axis=-1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)

    conv4 = Conv2D(np.int(256 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(np.int(256 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv4)
    bn4 = BatchNormalization(axis=-1)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)

    conv5 = Conv2D(np.int(512 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(np.int(512 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv5)
    bn5 = BatchNormalization(axis=-1)(conv5)

    up6 = Conv2D(np.int(256 * n_filters_factor),
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(
                     size=(2, 2), interpolation='nearest')(bn5))
    merge6 = concatenate([bn4, up6], axis=3)
    conv6 = Conv2D(np.int(256 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(np.int(256 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv6)
    bn6 = BatchNormalization(axis=-1)(conv6)

    up7 = Conv2D(np.int(256 * n_filters_factor),
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(
                     size=(2, 2), interpolation='nearest')(bn6))
    merge7 = concatenate([bn3, up7], axis=3)
    conv7 = Conv2D(np.int(256 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(np.int(256 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv7)
    bn7 = BatchNormalization(axis=-1)(conv7)

    up8 = Conv2D(np.int(128 * n_filters_factor),
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(
                     size=(2, 2), interpolation='nearest')(bn7))
    merge8 = concatenate([bn2, up8], axis=3)
    conv8 = Conv2D(np.int(128 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(np.int(128 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv8)
    bn8 = BatchNormalization(axis=-1)(conv8)

    up9 = Conv2D(np.int(64 * n_filters_factor),
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(
                     size=(2, 2), interpolation='nearest')(bn8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(np.int(64 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(np.int(64 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(np.int(64 * n_filters_factor),
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv9)

    final_layer_logits = [(Conv2D(n_output_classes, 1,
                                  activation='linear')(conv9))
                          for i in range(n_forecast_months)]
    final_layer_logits = tf.concat(final_layer_logits, axis=-1)

    model = Model(inputs, final_layer_logits)

    model.compile(optimizer=Adam(lr=learning_rate),
                  loss=loss,
                  weighted_metrics=weighted_metrics)
    model.summary()
    return model


if __name__ == '__main__':
    unet_batchnorm(input_shape=(112, 112, 7), loss='mse', weighted_metrics=0.1)
