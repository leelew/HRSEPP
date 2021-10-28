import argparse
import glob
import json
import re
import time

import config
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tensorflow as tf
import wandb
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import (EarlyStopping, LearningRateScheduler,
                                        ModelCheckpoint)
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from wandb.keras import WandbCallback

from model.unet import unet_batchnorm
from model.convlstm_wandb import convlstm_wandb


wandb.init()
# Metric to monitor for model checkpointing
mcMonitor = 'val_acc_mean'
mcMode = 'max'
esPatience = 10
esPatienceTransfer = np.inf  # No early stopping for the two transfer learning epochs

# Metric to monitor for early stopping
esMonitor = 'val_acc_mean'
esMode = 'max'

network_path = '/hard/lilu/Checkpoints/'


def make_exp_decay_lr_schedule(rate,
                               start_epoch=1,
                               end_epoch=np.inf,
                               verbose=False):
    ''' Returns an exponential learning rate function that multiplies by
    exp(-rate) each epoch after `start_epoch`. '''
    def lr_scheduler_exp_decay(epoch, lr):
        ''' Learning rate scheduler for fine tuning.
        Exponential decrease after start_epoch until end_epoch. '''

        if epoch >= start_epoch and epoch < end_epoch:
            lr = lr * np.math.exp(-rate)

        if verbose:
            print('\nSetting learning rate to: {}\n'.format(lr))

        return lr

    return lr_scheduler_exp_decay


obs_callbacks = []

obs_callbacks.append(
    ModelCheckpoint(network_path,
                    monitor=mcMonitor,
                    mode=mcMode,
                    verbose=1,
                    save_best_only=True))

obs_callbacks.append(
    EarlyStopping(monitor=esMonitor,
                  mode=esMode,
                  verbose=1,
                  patience=esPatience))

obs_callbacks.append(
    WandbCallback(monitor=mcMonitor,
                  mode=mcMode,
                  log_weights=False,
                  log_gradients=False))

lr_schedule = LearningRateScheduler(
    make_exp_decay_lr_schedule(
        rate=0.1,
        start_epoch=3,  # Start reducing LR after 3 epochs
        end_epoch=np.inf,
    ))
obs_callbacks.append(lr_schedule)
mask = np.load('mask.npy')[:112,:112]
model = convlstm_wandb(#unet_batchnorm(
    input_shape=(1, 112, 112, 8),
    loss=0,
    mask = mask,
    weighted_metrics=0.1,
    learning_rate=wandb.config.learning_rate,
    filter_size=wandb.config.filter_size,
    n_filters_factor=wandb.config.n_filters_factor,
    n_forecast_months=1,
    use_temp_scaling=False,
    n_output_classes=1,
)
x_train = np.load('x_train_sub.npy')
y_train = np.load('y_train_sub.npy')

model.fit(x_train[:, np.newaxis],
          y_train,
          batch_size=32,
          epochs=50,
          callbacks=obs_callbacks,
          validation_split=0.2)
y_pred = model.predict(x_train[:, np.newaxis])
from sklearn.metrics import r2_score

r2 = np.full((112,112), np.nan)

for i in range(112):
    for j in range(112):
        r2[i,j] = r2_score(y_train[:, i, j], y_pred[:, i, j])

        print(r2_score(y_train[:, i, j], y_pred[:, i, j]))
print(mask.shape)
print(r2.shape)
r2 = np.multiply(np.squeeze(r2),mask[:112,:112])

np.save('r2_convlstm.npy', r2)
