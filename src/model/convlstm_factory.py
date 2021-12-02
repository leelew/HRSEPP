import numpy as np
import tensorflow as tf
from factory.layers import DIConvLSTM2D
from tensorflow.keras import Input, Model, backend, layers
from tensorflow.keras.layers import (Add, BatchNormalization, Conv2D,
                                     ConvLSTM2D, Dense, Dropout,
                                     GlobalAveragePooling2D, Input, Lambda,
                                     MaxPooling2D, ReLU, Reshape, UpSampling2D,
                                     concatenate, multiply)
from tensorflow.keras.models import Model

def base_model():
    inputs = Input(shape=(7, 112, 112, 8))
    out = tf.keras.layers.ConvLSTM2D(8, 5, 
                        return_sequences=True, 
                        padding='same', 
                        kernel_initializer='he_normal')(inputs)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)
    mdl = Model(inputs, out)

    return mdl


def SMNet():
    # inputs 
    in_l3 = Input(shape=(7, 112, 112, 1))
    in_l4 = Input(shape=(7, 112, 112, 8))
    
    # preprocess l3
    out_l3 = DIConvLSTM2D.DIConvLSTM(filters=8, kernel_size=5)(in_l3)
    out_l3 = tf.keras.layers.BatchNormalization()(out_l3)
    out_l3 = tf.keras.layers.ReLU()(out_l3)

    # preprocess l4
    base_model()
    out_l4 = tf.keras.layers.ConvLSTM2D(8, 5, 
                        return_sequences=True, 
                        padding='same', 
                        kernel_initializer='he_normal')(in_l4)
    out_l4 = tf.keras.layers.BatchNormalization()(out_l4)
    out_l4 = tf.keras.layers.ReLU()(out_l4)

    out = tf.keras.layers.Add()([out_l3, out_l4, in_l4])

    states = ConvLSTM2D(16, 3, padding='same', return_sequences=True)(out)
    states = tf.transpose(states, [0, 4, 2, 3, 1])
    states = Dense(1)(states)
    states = tf.transpose(states, [0, 4, 2, 3, 1])

    out = Lambda(lambda x: backend.concatenate([x] * 7, axis=1))(states)
    out = ConvLSTM2D(16, 3, padding='same', return_sequences=True)(out)
    out = tf.keras.layers.Dense(1)(out)

    mdl = Model([in_l3, in_l4], out)
    mdl.summary()

    return mdl
