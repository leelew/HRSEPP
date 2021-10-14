import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
import torch

class GlobalLSTM(Model):
    def __init__(self, hidden_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.hidden_size = hidden_size
        self.in_layer = layers.Dense(hidden_size, activation='relu')
        self.hidden_layer = layers.LSTM(hidden_size, return_sequences=True)
        self.out_layer = layers.Dense(1, activation=None)
    
    def call(self, X):
        x = self.in_layer(X)
        x = self.hidden_layer(x)
        x = self.out_layer(x)
        return x

def randomIndex(ngrid, nt, dimSubset, bufftime=0):
    batchSize, rho = dimSubset
    iGrid = np.random.randint(0, ngrid, [batchSize])
    iT = np.random.randint(0+bufftime, nt - rho, [batchSize])
    return iGrid, iT


def selectSubset(x, iGrid, iT, rho, *, c=None, tupleOut=False, LCopt=False, bufftime=0):
    nx = x.shape[-1]
    nt = x.shape[1]
    if x.shape[0] == len(iGrid):   #hack
        iGrid = np.arange(0,len(iGrid))  # hack
    if (rho is not None) and (nt <= rho):
        iT.fill(0)

    batchSize = iGrid.shape[0]

    # batchSize = iGrid.shape[0]
    xTensor = np.zeros([rho+bufftime, batchSize, nx])
    for k in range(batchSize):
        temp = x[iGrid[k]:iGrid[k] + 1, np.arange(iT[k]-bufftime, iT[k] + rho), :]
        xTensor[:, k:k + 1, :] = np.swapaxes(temp, 1, 0)
        
    return xTensor


class RMSELoss(tf.keras.losses.Loss):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def call(self, output, target):
        ny = target.shape[2]
        loss = 0
        for k in range(ny):
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            temp = tf.math.sqrt(tf.math.reduce_mean((p - t)**2))
            loss = loss + temp
        return loss

def train(model, X, y, loss_func, epochs=500, mini_batch=[100, 30], bufftime=0):

    # get parameters
    num_grid, n_t, n_x = X.shape
    batch_size, rho = mini_batch

    # batch size larger than total grids
    if batch_size >= num_grid: batch_size = num_grid

    # auto setting nIter
    nIter = int(np.ceil(np.log(0.01) / np.log(1 - batch_size * rho / num_grid / (n_t-bufftime))))

    # set optimizer, NOTE: the same setting with torch.optim.Adadelta
    optimizer = tf.keras.optimizers.Adadelta(learning_rate=1, rho=0.9, epsilon=1e-6)

    # train
    for i in range(1, epochs + 1):
        loss_epoch = 0

        t0 = time.time()
        for _ in range(0, nIter):
            
            # make training data by sampling
            iGrid, iT = randomIndex(num_grid, n_t, [batch_size, rho])
            x_train = selectSubset(X, iGrid, iT, rho)
            y_train = selectSubset(y, iGrid, iT, rho)

            # tape gradient

            with tf.GradientTape() as tape:
                y_pred = model(x_train)
                loss = loss_func(y_pred, y_train)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables)) 
            
            loss_epoch += loss
        t1 = time.time()
        
        # print log
        print("epoch: {}, loss: {}, time: {}".format(i, loss_epoch/nIter, t1-t0))
        
    return model












"""
class GlobalLSTMCell(keras.layers.Layer):

    def __init__(self, hidden_size):
        super().__init__()

        # share parameters for all timesteps
        self._ifo_x = keras.layers.Dense(3 * hidden_size, activation=None)
        self._ifo_h = keras.layers.Dense(3 * hidden_size, activation=None)
        self._b_x = keras.layers.Dense(hidden_size, activation=None)
        self._b_h = keras.layers.Dense(hidden_size, activation=None)

    def call(self, inputs, h_prev=None, c_prev=None):
        #default lstm pass
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
        #
        #x0 = self.in_layer(X)
        #print(x0.shape)
        out_lstm = self.hidden_layer(X)
        
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
        
        out = self.out_layer(out_lstm)

        return out
    
"""