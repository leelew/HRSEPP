
import tensorflow as tf


def convlstm(input_len, n_lat, n_lon, n_feature):

    # inputs
    inputs = tf.keras.layers.Input(shape=(input_len, n_lat, n_lon, n_feature))

    x = tf.keras.layers.ConvLSTM2D(
        filters=128, kernel_size=(3, 3),
        padding='same', return_sequences=True,
        activation='tanh', recurrent_activation='hard_sigmoid',
        kernel_initializer='glorot_uniform', unit_forget_bias=True,
        dropout=0.3, recurrent_dropout=0.3, go_backwards=True)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ConvLSTM2D(
        filters=64, kernel_size=(3, 3),
        padding='same', return_sequences=True,
        activation='tanh', recurrent_activation='hard_sigmoid',
        kernel_initializer='glorot_uniform', unit_forget_bias=True,
        dropout=0.3, recurrent_dropout=0.3, go_backwards=True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ConvLSTM2D(
        filters=1, kernel_size=(3, 3),
        padding='same', return_sequences=False,
        activation='tanh', recurrent_activation='hard_sigmoid',
        kernel_initializer='glorot_uniform', unit_forget_bias=True,
        dropout=0.3, recurrent_dropout=0.3, go_backwards=True)(x)

    # build
    model = tf.keras.models.Model(inputs=inputs, outputs=x)

    # summary
    model.summary()

    return model
