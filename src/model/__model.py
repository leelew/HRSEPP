import numpy as np
import tensorflow as tf
from tensorflow import keras


class GlobalLSTMCell(keras.layers.Layer):

    """Share-weight LSTM

    Args:
    -----
    h, c: shape as (batch_size, hidden_size)
    x: shape as (batch_size, num_features)

    out:
    ----
    h, c: shape as (batch_size, hidden_size)
    y: shape as (batch_size, hidden_size)
    """
    def __init__(self, hidden_size):
        super().__init__()

        # share parameters for all timesteps
        self._ifo_x = keras.layers.Dense(3 * hidden_size, activation=None)
        self._ifo_h = keras.layers.Dense(3 * hidden_size, activation=None)
        self._b_x = keras.layers.Dense(hidden_size, activation=None)
        self._b_h = keras.layers.Dense(hidden_size, activation=None)

    def call(self, inputs, h_prev=None, c_prev=None):
        """default lstm pass"""
        # generate input, forget and output gates
        ifo = tf.sigmoid(self._ifo_x(inputs) + self._ifo_h(h_prev))
        i, f, o = tf.split(ifo, 3, axis=-1)

        #print(ifo.shape)

        # generate current information state
        a = tf.math.tanh(self._b_x(inputs) + self._b_h(h_prev))

        # generate current cell state
        c = tf.math.multiply(i, a) + tf.math.multiply(f, c_prev)

        # generate current hidden state
        h = tf.math.multiply(o, tf.math.tanh(c))

        return h, c 
       


class GlobalLSTM(keras.Model):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.in_layer = keras.layers.Dense(hidden_size, activation='relu')
        #self.hidden_layer = GlobalLSTMCell(hidden_size) 
        self.hidden_layer = keras.layers.LSTM(hidden_size, return_sequences=True)
        self.out_layer = keras.layers.Dense(1)

    def update_state(self, state, state_new, state_idx, N):

        if state_idx == 0:
            state = tf.concat([state_new, state[:, state_idx+1:, :]], axis=1)
        elif state_idx == N:
            state = tf.concat([state[:, :state_idx, :], state_new], axis=1)
        else:
            state = tf.concat(
                [state[:, :state_idx, :], state_new, state[:, state_idx+1:, :]], axis=1)

        return state

    def call(self, X, h0=None, c0=None):
        """
        Args
        ----
            X: the shape of inputs is (batch_size, timestep, hidden_size)
            the shape of outputs is (ngrids, nt, 1)
        """
        #x0 = self.in_layer(X)
        #print(x0.shape)
        out_lstm = self.hidden_layer(X)
        """
        batch_size, timesteps, num_features = X.get_shape().as_list()

        if h0 is None:
            h0 = tf.zeros([batch_size, 1, self.hidden_size])
        if c0 is None:
            c0 = tf.zeros([batch_size, 1, self.hidden_size])
        

        #print(h0.shape)

        out_lstm = tf.zeros_like(x0)
        for i in range(timesteps):
            h0, c0 = self.hidden_layer(x0[:,i,:][:, tf.newaxis,:], h0, c0)
            #print(h0.shape)
            out_lstm = self.update_state(out_lstm, h0, i, timesteps)
        """
        out = self.out_layer(out_lstm)
        
        return out
    
    
 


if __name__ == '__main__':

    x = np.zeros([100, 30, 19])
    out = GlobalLSTM(256)(x)
    print(out.shape)



            



