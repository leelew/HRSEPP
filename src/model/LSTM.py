


import tensorflow as tf



def LSTM(n_feature, input_len):

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