from os import name
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error

def random_index(num_grid, n_t, batch_size, rho, bufftime=0):

    if num_grid < 100: batch_size = num_grid

    idx_grid = np.random.randint(0, num_grid, [batch_size])
    idx_t = np.random.randint(0+bufftime, n_t - rho, [batch_size])
    
    return idx_grid, idx_t


def select_subset(x, idx_grid, idx_t, rho, bufftime=0):

    x_subset = np.zeros([len(idx_grid),rho+bufftime,x.shape[-1]])
    for k in range(len(idx_grid)):
        tmp = x[idx_grid[k]:idx_grid[k]+1, np.arange(idx_t[k]-bufftime, idx_t[k]+rho), :]
        x_subset[k:k+1, :, :] = tmp

    return x_subset


def rmse_loss(y_true, y_pred):
    return tf.math.reduce_sum(tf.math.sqrt((y_true-y_pred)**2))

def train(model,
          X, 
          y,
          loss_func,
          epochs=500,
          batch_size=6,
          rho=30
          ):

    optimizer = tf.keras.optimizers.Adam()
    num_grid, n_t, n_f = X.shape
    for i_epoch in range(epochs):
        for i_iter in range(230):

            with tf.GradientTape() as tape:
                idx_grid, idx_t = random_index(num_grid, n_t, batch_size, rho, bufftime=0)

                x_train = select_subset(X, idx_grid, idx_t, rho, bufftime=0)
                y_train = select_subset(y, idx_grid, idx_t, rho, bufftime=0)

                y_p = model(x_train)
                #print(y_p.shape)
                #print(y_train.shape)
                #loss = tf.keras.losses.mean_squared_error(y_train, y_p)
                loss = loss_func(y_train, y_p)
                #print(loss)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables)) 

        print("epoch {}: MSE is {}".format(i_epoch, loss))

    return model   


if __name__ == '__main__':
    from __model import GlobalLSTM
    model = GlobalLSTM(256)

    X = np.random.random([6, 718, 7])
    y = np.random.random([6, 718, 1])


    model = train(model,
          X, 
          y,
          loss_func=rmse_loss,
          epochs=1,
          batch_size=6,
          rho=30)
    
    X_test = np.random.random([100, 365, 7])
    y_ = model(X_test)
    print(y_.shape)
