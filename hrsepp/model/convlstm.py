import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (BatchNormalization, Conv2D, ConvLSTM2D,
                                     Dense, Input, MaxPooling2D, UpSampling2D,
                                     concatenate, Lambda)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers.core import Dense
from tensorflow.keras import backend
from utils.loss import MaskMSELoss, MaskSSIMLoss


def ed_convlstm(input_shape,
                len_output,
                mask,
                learning_rate=1e-4,
                filter_size=3,
                n_filters_factor=1):
    inputs = Input(shape=input_shape)

    c = ConvLSTM2D(np.int(16 * n_filters_factor),
                   filter_size,
                   return_sequences=False,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(inputs)
    c = BatchNormalization(axis=-1)(c)

    c = Lambda(lambda x: backend.concatenate([x[:, np.newaxis]] * len_output,
                                             axis=1))(c)

    c = ConvLSTM2D(np.int(16 * n_filters_factor),
                   filter_size,
                   return_sequences=True,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(c)

    out = Dense(1, activation=None)(c)
    model = Model(inputs, out)

    model.compile(optimizer=Adam(lr=learning_rate), loss=MaskMSELoss(mask))
    model.summary()
    return model

def convlstm1(input_shape,
              mask,
              learning_rate=1e-4,
              filter_size=3,
              n_filters_factor=1):
    inputs = Input(shape=input_shape)

    c1 = ConvLSTM2D(np.int(16 * n_filters_factor),
                    filter_size,
                    return_sequences=True,
                    activation='relu',
                    padding='same',
                    kernel_initializer='he_normal')(inputs)
    bn1 = BatchNormalization(axis=-1)(c1)

    out = Dense(1, activation=None)(bn1)

    model = Model(inputs, out)

    model.compile(optimizer=Adam(lr=learning_rate), loss=MaskMSELoss(mask))
    model.summary()

    return model


def convlstm3(input_shape,
              mask,
              learning_rate=1e-4,
              filter_size=3,
              n_filters_factor=1):
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
    convlstm3(input_shape=(1, 112, 112, 8), loss='mse')
