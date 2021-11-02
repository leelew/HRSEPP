import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (EarlyStopping, LearningRateScheduler,
                                        ModelCheckpoint)
from wandb.keras import WandbCallback


class CallBacks():
    callbacks = []

    def __init__(
            self,
            # Metric to monitor for model checkpointing
            mcMonitor='val_acc_mean',
            mcMode='max',
            esPatience=10,
            # Metric to monitor for early stopping
            esMonitor='val_acc_mean',
            esMode='max',
            network_path='/hard/lilu/Checkpoints/') -> None:
        self.mcMonitor = mcMonitor
        self.mcMode = mcMode
        self.esPatience = esPatience
        self.esMonitor = esMonitor
        self.esMode = esMode
        self.network_path = network_path

    def __call__(self):

        self.callbacks.append(
            ModelCheckpoint(self.network_path,
                            monitor=self.mcMonitor,
                            mode=self.mcMode,
                            verbose=1,
                            save_best_only=True))

        self.callbacks.append(
            EarlyStopping(monitor=self.esMonitor,
                          mode=self.esMode,
                          verbose=1,
                          patience=self.esPatience))

        self.callbacks.append(
            WandbCallback(monitor=self.mcMonitor,
                          mode=self.mcMode,
                          log_weights=False,
                          log_gradients=False))

        self.callbacks.append(
            LearningRateScheduler(
                self.make_exp_decay_lr_schedule(
                    rate=0.1,
                    start_epoch=3,  # Start reducing LR after 3 epochs
                    end_epoch=np.inf,
                )))

    @staticmethod
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
