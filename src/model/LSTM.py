


import tensorflow as tf



def LSTM(n_feature, input_len):

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_visible_devices(devices=gpus[1], device_type='GPU')
        tf.config.experimental.set_memory_growth(device=gpu, enable=True)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():

        # inputs
        inputs = tf.keras.layers.Input(shape=(input_len, n_feature))
        x = tf.keras.layers.LSTM(units=256, return_sequences=True)(inputs)
        x = tf.keras.layers.LSTM(units=256, return_sequences=False)(x)
        x = tf.keras.layers.Dense(1)(x)

        # build
        model = tf.keras.models.Model(inputs=inputs, outputs=x)
        
        # summary
        model.summary()

    return model