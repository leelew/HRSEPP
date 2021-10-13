import tensorflow as tf
from tensorflow import keras
import numpy as np



class GlobalLSTMCell(keras.layers):
    """Share-weight LSTM
    """
    def __init__(self, input_size, hidden_size,) -> None:
        super().__init__()

        # share parameters for all timesteps
        self.w_ih = tf.Tensor(hidden_size * 4, input_size)
        self.w_hh = tf.Tensor(hidden_size * 4, input_size)
        self.b_ih = tf.Tensor(hidden_size * 4)
        self.b_hh = tf.Tensor(hidden_size * 4)

        self.
        #  

    def call(self, inputs, hx=None, cx=None):
        batch_size = inputs.shape[0]

        if hx is None:
            hx = tf.zeros(1, self.batch_size, self.hidden_size)
        if cx is None:
            cx = tf.zeros(1, self.batch_size, self.hidden_size)

        




class GlobalLSTM(keras.Model):
    def __init__(self, *, hidden_size) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.in_layer = keras.layers.Dense(hidden_size, activation='ReLU')
        self.hidden_layer = keras.layers.LSTMCell(hidden_size)
        self.out_layer = keras.layers.Dense(1)

    def call(self, X):
        """
        Args
        ----
            X: the shape of inputs is (ngrids, nt, nf)
            the shape of outputs is (ngrids, nt, 1)
        """
        x0 = self.in_layer(X)
        batch_size = X.shape[0]
        timestep = X.shape[1]
        nf = X.shape[2]

        for i in range(timestep):
            outLSTM[:, i, :], (h, c) = self.hidden_layer(x0)
        
        return self.out_layer(outLSTM)

def train(model,
          X, 
          y,
          loss_func,
          epochs=500,
          batch_size=100,
          iters,
          ):

    
    yP = model(X)
    loss = loss_func(y, yP)

    for i_epoch in range(epochs):
        for i_iter in range(iters):
            
            



