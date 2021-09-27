import tensorflow as tf
import numpy as np

class trainer():

    def __init__(self, 
                 ngrid, 
                 batch_size, 
                 loss_fun, 
                 optim,
                 rho,
                 learning_rate,
                 epochs):
        
        self.ngrid = ngrid
        self.batch_size = batch_size
        self.optim = optim
        self.rho = rho
        self.lr = learning_rate
        self.epochs = epochs
        self.loss_fun = loss_fun


    def train_step(self, model, X, y, optimizer):

        with tf.GradientTape() as tape:
            idx = self.random_idx(self.ngrid, self.batch_size)
            x_train = self.select_subset(X, idx)
            y_train = self.select_subset(y, idx)

            y_predict = model(x_train)

            loss = self.loss_fun(y_predict, y_train)

            grad = tape.gradient(loss, model.trainable_variables)

            optimizer.apply_gradients(zip(grad, model.trainable_variables))

    def fit(
            model,
            X, 
            y, 
            lossFun,
            ngrid, 
            epochs,
            optim,
            rho,  
            lr,
            batch_size,):

        if X.ndims != 5 or y.ndims != 4:
            raise ValueError('pls inputs correct X, y')
        else:
            Nx, Nt, Nlat, Nlon, Nf = X.shape

        if batch_size >= ngrid: 
            batch_size = ngrid
        
        if optim == 'adadelta':
            optimizer = tf.keras.optimizers.Adadelta(learning_rate=lr, rho=rho)
        elif optim == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    @staticmethod
    def random_idx(ngrid, batch_size):
        idx = np.random.randint(0, ngrid, [batch_size])
        return idx


    @staticmethod
    def select_subset(input, idx):
        if input.ndims == 4:
            

        return input[:, ]
        

    @staticmethod
    def save_model():
        pass