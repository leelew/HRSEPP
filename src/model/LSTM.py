


import tensorflow as tf



def LSTM():

    # inputs
    inputs = tf.keras.layers.Input(shape=(10, 6))
    x = tf.keras.layers.LSTM(units=1, return_sequences=False)(inputs)

    # build
    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    
    # summary
    model.summary()

    return model
