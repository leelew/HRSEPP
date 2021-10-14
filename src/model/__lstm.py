import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
import torch


class LSTM(Model):
    def __init__(self, hidden_size, output_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.hidden_size = hidden_size
        self.hidden_layer = layers.LSTM(hidden_size, return_sequences=True)
        self.out_layer = layers.Dense(output_size, activation=None)
    
    def call(self, X):
        # X: (713, 1, 42)
        # y: (713, 1, 1)
        x = self.hidden_layer(X)
        x = self.out_layer(x)
        return x


def train(model, x, y):
    model.compile(loss='mse', optimizer='adam')
    model.fit(x, y, batch_size=128, epochs=200)

    return model

