

import tensorflow as tf

"""
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_visible_devices(
        devices=gpus[1], device_type='GPU')
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
"""


def lstm():

    # inputs
    inputs = tf.keras.layers.Input(shape=(1, 42))
    x = tf.keras.layers.LSTM(units=256, return_sequences=True)(inputs)
    x = tf.keras.layers.Dense(6)(x)

    # build
    model = tf.keras.models.Model(inputs=inputs, outputs=x)

    # summary
    model.summary()

    return model
