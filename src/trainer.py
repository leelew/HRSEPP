import tensorflow as tf
import numpy as np
import tqdm



def keras_train(model, X, y, batch_size, epochs, save_folder):

    model.compile(loss='mse', optimizer='adam')
    model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
    model.save(save_folder)

    return model



class trainer():

    def __init__(self, 
                 ngrid, 
                 batch_size, 
                 loss_fun, 
                 optim,
                 rho,
                 learning_rate,
                 epochs):
        
        if batch_size >= ngrid: 
            batch_size = ngrid

        self.ngrid = ngrid
        self.batch_size = batch_size
        self.optim = optim
        self.rho = rho
        self.lr = learning_rate
        self.epochs = epochs
        self.loss_fun = loss_fun

        if self.optim == 'adadelta':
            self.optimizer = tf.keras.optimizers.Adadelta(learning_rate=self.lr, rho=self.rho)
        elif self.optim == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def fit(self,
            model,
            X, 
            y,
            save_folder=None):

        if X.ndims != 5 or y.ndims != 4:
            raise ValueError('pls inputs correct X, y')
        else:
            Nx, Nt, Nlat, Nlon, Nf = X.shape

        # according to setting of hydroDL from mhpi
        nIter = int(np.ceil(np.log(0.01) / np.log(1 - self.batch_size * 30 / self.ngrid / Nt)))

        for epoch in tqdm(range(self.epochs)):

            for iter in range(nIter):
                with tf.GradientTape() as tape:

                    # select X, y
                    idx = self.random_idx(self.ngrid, self.batch_size)
                    x_train = self.select_subset(X, idx)
                    y_train = self.select_subset(y, idx)

                    # generate loss
                    y_predict = model(x_train)
                    loss = self.loss_fun(y_predict, y_train)

                    # apply gradient
                    grad = tape.gradient(loss, model.trainable_variables)
                    self.optimizer.apply_gradients(zip(grad, model.trainable_variables))




        return model


                    

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