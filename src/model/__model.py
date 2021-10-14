import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers


class GlobalLSTM(Model):
    def __init__(self, hidden_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.hidden_size = hidden_size
        self.in_layer = layers.Dense(hidden_size, activation='relu')
        self.hidden_layer = layers.LSTM(hidden_size, return_sequences=True)
        self.out_layer = layers.Dense(1)
    
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
    if iT is not None:
        # batchSize = iGrid.shape[0]
        xTensor = tf.zeros([rho+bufftime, batchSize, nx], requires_grad=False)
        for k in range(batchSize):
            temp = x[iGrid[k]:iGrid[k] + 1, np.arange(iT[k]-bufftime, iT[k] + rho), :]
            xTensor[:, k:k + 1, :] = tf.Tensor(np.swapaxes(temp, 1, 0))
    else:
        if LCopt is True:
            # used for local calibration kernel: FDC, SMAP...
            if len(x.shape) == 2:
                # Used for local calibration kernel as FDC
                # x = Ngrid * Ntime
                xTensor = torch.from_numpy(x[iGrid, :]).float()
            elif len(x.shape) == 3:
                # used for LC-SMAP x=Ngrid*Ntime*Nvar
                xTensor = torch.from_numpy(np.swapaxes(x[iGrid, :, :], 1, 2)).float()
        else:
            # Used for rho equal to the whole length of time series
            xTensor = torch.from_numpy(np.swapaxes(x[iGrid, :, :], 1, 0)).float()
            rho = xTensor.shape[0]
    if c is not None:
        nc = c.shape[-1]
        temp = np.repeat(
            np.reshape(c[iGrid, :], [batchSize, 1, nc]), rho+bufftime, axis=1)
        cTensor = torch.from_numpy(np.swapaxes(temp, 1, 0)).float()

        if (tupleOut):
            if torch.cuda.is_available():
                xTensor = xTensor.cuda()
                cTensor = cTensor.cuda()
            out = (xTensor, cTensor)
        else:
            out = torch.cat((xTensor, cTensor), 2)
    else:
        out = xTensor

    if torch.cuda.is_available() and type(out) is not tuple:
        out = out.cuda()
    return out


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



            



