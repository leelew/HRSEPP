import sys

sys.path.append('../../hrsepp/')

import json

import numpy as np
import tensorflow as tf
from data.data_loader import DataLoader
from model.convlstm import convlstm1, convlstm3, ed_convlstm, seq2seq_convlstm
from model.seq2seq import seq2seq_attn_ConvLSTM
from model.unet import unet5, unet9
from sklearn.metrics import r2_score
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from utils.callback import CallBacks
from model.causality_tree import CausalTree
import wandb
from model.causal_convlstm import causal_convlstm

def train(x_train,
          y_train,
          x_valid,
          y_valid,
          x_test,
          y_test,
          x_train_dec,
          x_valid_dec,
          x_test_dec,
          mask,
          ID,
          input_shape,
          learning_rate,
          n_filters_factor,
          filter_size,
          batch_size,
          epochs,
          n_forecast_months,
          n_output_classes,
          model_name,
          save_path,
          wanda_mode=False):
    print('mask shape is {}'.format(mask.shape))
    # wandb setting
    print('n_factor is {}'.format(n_filters_factor))
    default = dict(learning_rate=learning_rate,
                   n_filters_factor=n_filters_factor,
                   filter_size=filter_size,
                   batch_size=batch_size,
                   epochs=epochs)

    wandb.init(config=default, allow_val_change=True)
    inputs = np.nanmean(x_train,axis=(-2, -3))[:,-1]
    child = CausalTree()(inputs)

    print(child)
    # model
    #model = ed_convlstm(input_shape=(1, 224, 224, 8),
    #                    len_output=7,
    #                    mask=mask,
    #                    learning_rate=wandb.config.learning_rate,
    #                    filter_size=wandb.config.filter_size,
    #                    n_filters_factor=wandb.config.n_filters_factor)
    if model_name == 'convlstm':
        model = convlstm1(input_shape=input_shape,
                          mask=mask,
                          learning_rate=wandb.config.learning_rate,
                          filter_size=wandb.config.filter_size,
                          n_filters_factor=wandb.config.n_filters_factor)
    elif model_name == 'seq2seq':
        print(wandb.config.n_filters_factor)
        model = seq2seq_convlstm(
            input_shape=input_shape,
            len_output=7,
            mask=mask,
            learning_rate=wandb.config.learning_rate,
            filter_size=wandb.config.filter_size,
            n_filters_factor=wandb.config.n_filters_factor)
    elif model_name == 'causal':
        model = causal_convlstm(input_shape=input_shape,
                          child=child,
                          mask=mask,
                          learning_rate=wandb.config.learning_rate,
                          kernel_size=wandb.config.filter_size,
                          n_filters_factor=wandb.config.n_filters_factor)
        #model = seq2seq_attn_ConvLSTM(
        #    mask=mask,
        #    learning_rate=wandb.config.learning_rate)


    else:
        model = unet9(input_shape=input_shape,
                      mask=mask,
                      learning_rate=wandb.config.learning_rate,
                      filter_size=wandb.config.filter_size,
                      n_filters_factor=wandb.config.n_filters_factor,
                      n_forecast_months=n_forecast_months,
                      n_output_classes=n_output_classes)

    if model_name == 'cnn':
        x_train = x_train[:, 0]
        x_valid = x_valid[:, 0]
        y_train = y_train[:, 0]
        y_valid = y_valid[:, 0]
        x_test = x_test[:, 0]

    np.save('/hard/lilu/y_train_obs_{}.npy'.format(ID), y_train)
    np.save('/hard/lilu/y_valid_obs_{}.npy'.format(ID), y_valid)
    np.save('/hard/lilu/y_test_obs_{}.npy'.format(ID), y_test)
    np.save('/hard/lilu/x_train_obs_{}.npy'.format(ID), x_train)
    np.save('/hard/lilu/x_valid_obs_{}.npy'.format(ID), x_valid)
    np.save('/hard/lilu/x_test_obs_{}.npy'.format(ID), x_test)

    from IO.mixup import augment

    train_ds, val_ds = augment(x_train,
                               y_train,
                               x_valid,
                               y_valid,
                               BATCH_SIZE=wandb.config.batch_size)
    print(x_train_dec.shape) 
    print(x_train.shape)  
    print(y_train.shape) 
    model.fit(
        [x_train, x_train_dec],
        y_train,
        batch_size=wandb.config.batch_size,
        epochs=wandb.config.epochs,
        callbacks=CallBacks()(),
        #validation_split=0.2)
        validation_data=([x_valid, x_valid_dec], y_valid))
    model.load_weights('/hard/lilu/Checkpoints/') 
    #y_train_pred = model.predict([x_train, x_train_dec])
    #np.save('/hard/lilu/y_train_pred_{}'.format(ID), y_train_pred)
    del x_train, x_train_dec

    #y_valid_pred = model.predict([x_valid, x_valid_dec])
    #np.save('/hard/lilu/y_valid_pred_{}'.format(ID), y_valid_pred)
    del x_valid,  x_valid_dec

    y_test_pred = model.predict([x_test, x_test_dec])
    np.save('/hard/lilu/y_test_pred_{}'.format(ID), y_test_pred)
    del x_test, y_test_pred, x_test_dec

    model.save(save_path)


if __name__ == '__main__':

    x_train = np.load('/hard/lilu/x_train_2.npy')
    y_train = np.load('/hard/lilu/y_train_2.npy')

    x_test = np.load('/hard/lilu/x_test_2.npy')
    y_test = np.load('/hard/lilu/y_test_2.npy')

    x_tr, y_tr = DataLoader(len_input=1,
                            len_output=1,
                            window_size=2,
                            use_lag_y=True,
                            mode='train')(x_train, y_train)

    x_te, y_te = DataLoader(len_input=1,
                            len_output=1,
                            window_size=2,
                            use_lag_y=True,
                            mode='train')(x_test, y_test)
    print(x_tr.shape)
    print(y_tr.shape)
    print(x_te.shape)
    print(y_te.shape)

    np.save('y_train', y_tr)
    np.save('y_test', y_te)

    with open('/hard/lilu/SMAP_L4/test/auxiliary.json') as f:
        aux = json.load(f)

    train(x_tr[:, 0, :, :, :],
          y_tr[:, :, :, :, :],
          x_te[:, 0, :, :, :],
          y_te[:, :, :, :, :],
          np.array(aux['mask']),
          ID=2,
          model_name='ed_convlstm',
          save_path='/hard/lilu/SMAP_L4/model/unet/')
