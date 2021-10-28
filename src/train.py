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

model = unet_batchnorm(
    input_shape=(128, 128, 3),
    loss='mse',
    weighted_metrics=0.1,
    learning_rate=wandb.config.learning_rate,
    filter_size=wandb.config.filter_size,
    n_filters_factor=wandb.config.n_filters_factor,
    n_forecast_months=7,
    use_temp_scaling=False,
    n_output_classes=1,
)

model.fit(x_train,
          y_train,
          batch_size=2,
          epochs=50,
          callbacks=obs_callbacks,
          validation_split=0.2)
