import numpy as np
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.layers import (BatchNormalization, Conv2D, ConvLSTM2D,
                                     Dense, Dropout, GlobalAveragePooling2D,
                                     Input, Lambda, MaxPooling2D, ReLU,
                                     Reshape, UpSampling2D, concatenate,
                                     multiply)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from utils.loss import MaskMSELoss, MaskSSIMLoss

from layers.seq2seqConvLSTM import SimpleSeq2seqConvLSTM

#from model.saconvlstm import SaConvLSTM2D


def seq2seq_convlstm(input_shape,
                     len_output,
                     mask,
                     learning_rate=1e-4,
                     filter_size=3,
                     n_filters_factor=1):
    inputs = Input(shape=input_shape)
    c = SimpleSeq2seqConvLSTM(np.int(16 * n_filters_factor),
                              filter_size,
                              dec_len=len_output)(inputs)
    c = BatchNormalization(axis=-1)(c)
    out = Dense(1, activation=None)(c)
    model = Model(inputs, out)

    model.compile(optimizer=Adam(lr=learning_rate), loss=MaskMSELoss(mask))
    model.summary()
    return model


def ed_convlstm(input_shape,
                len_output,
                mask,
                learning_rate=1e-4,
                filter_size=3,
                n_filters_factor=1):
    inputs = Input(shape=input_shape)

    c = ConvLSTM2D(
        np.int(16 * n_filters_factor),
        filter_size,
        return_sequences=False,
        #activation='relu',
        padding='same',
        kernel_initializer='he_normal')(inputs)
    c = BatchNormalization(axis=-1)(c)
    c = ReLU(c)
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


class se_5d(tf.keras.layers.Layer):
    def __init__(self, input_shape):
        super().__init__()
        self.dense1 = Dense(input_shape[-1] // 8)
        #activation='relu',
        #kernel_initializer='he_normal',
        #use_bias=True,
        #bias_initializer='zeros')
        self.dense2 = Dense(input_shape[-1])
        #activation='relu',
        #kernel_initializer='he_normal',
        #use_bias=True,
        #bias_initializer='zeros')

        self.gap = GlobalAveragePooling2D()  #(keepdims=True)

    def call(self, inputs, input_shape):
        t, lat, lon, c = input_shape
        print((t, lat, lon, c))
        a = Reshape((lat, lon, c * t))(inputs)
        print(a.shape)
        se = self.gap(a)  # tf version > 2.6.0
        se = Reshape((1, 1, c * t))(se)
        print(se.shape)
        se = self.dense1(se)
        print(se.shape)
        se = self.dense2(se)
        print(se.shape)
        a = multiply([a, se])
        print(a.shape)
        inputs = Reshape((t, lat, lon, c))(a)

        return inputs


def convlstm1(input_shape,
              mask,
              learning_rate=1e-4,
              filter_size=3,
              n_filters_factor=1):
    inputs = Input(shape=input_shape)
    #c1 = BatchNormalization(axis=-1)(inputs)
    #c1 = Dropout(0.05)(c1)
    #x = se_5d(input_shape)(inputs, input_shape)

    #x = Dense(2)(inputs)
    #x = Dense(input_shape[-1])(x)
    x = inputs
    c1 = ConvLSTM2D(
        np.int(16 * n_filters_factor),
        filter_size,
        return_sequences=True,
        #activation='relu',
        #activation='linear', adopt 'tanh'.
        padding='same',
        #use_bias=False, # remove bias before BN could increase perform
        kernel_initializer='he_normal')(x)
    bn1 = BatchNormalization(axis=-1)(c1)
    bn1 = tf.nn.relu(
        bn1
    )  # BN before ReLU. advised by Jinjing Pan according to ResNet setting.
    out = Dense(1, activation=None)(c1)

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
