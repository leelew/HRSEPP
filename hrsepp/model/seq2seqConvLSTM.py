import tensorflow as tf


class encoder(tf.keras.layers.Layer):
    def __init__(self) -> None:
        super().__init__()

        self.convlstm = tf.keras.layers.ConvLSTM2D()

    def call(self, X_encoder):
        out, state_h, state_c = self.convlstm(X,
                                              return_state=True,
                                              return_sequence=True)

        return out, [state_h, state_c]


class decoder(tf.keras.layers.Layer):
    def __init__(self) -> None:
        super().__init__()
        self.convlstm = tf.keras.layers.ConvLSTM2D()

    def call(self, X_decoder, initial_state):
        self.convlstm(X,
                      return_state,
                      return_sequence=True,
                      initial_state=initial_state)
