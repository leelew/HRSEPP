# ==============================================================================
# ConvLSTM in seq2seq mode
#
# (1) simple:
# (2) teacher forcing:
#
# author: Lu Li
# email: lilu83@mail.sysu.edu.cn
# ==============================================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.python.keras.layers.convolutional_recurrent import ConvLSTM2DCell


class Seq2seqConvLSTM(layers):
    def __init__(self,
                 n_filters_factor,
                 filter_size,
                 dec_len,
                 train_mode=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.dec_len = dec_len
        self.train_mode = None

        # encode convlstm
        self.enc_convlstm = layers.ConvLSTM2D(np.int(16 * n_filters_factor),
                                              kernel_size=filter_size,
                                              return_sequences=True,
                                              return_state=True,
                                              padding='same',
                                              kernel_initializer='he_normal')
        self.enc_bn = layers.BatchNormalization(axis=-1)

        # decode convlstm
        self.dec_convlstm = layers.ConvLSTM2D(np.int(16 * n_filters_factor),
                                              kernel_size=filter_size,
                                              return_sequences=True,
                                              padding='same',
                                              kernel_initializer='he_normal')
        self.dec_bn = layers.BatchNormalization(axis=-1)

        #
        self.dec_convlstm_cell = ConvLSTM2DCell(np.int(16 * n_filters_factor),
                                                kernel_size=filter_size,
                                                padding='same',
                                                kernel_initializer='he_normal')

    def call(self, x_encoder, x_decoder=None, training=None, mask=None):

        enc_out, state = self.encoder(x_encoder=x_encoder)

        if self.train_mode == 'teacher':
            if x_decoder:
                raise KeyError('teacher forcing training needs decoder inputs')
            dec_out = self.decoder(x_decoder, initial_state=state)
        else:
            dec_out = self.decoder(enc_out[:, -1], initial_state=state)
        return dec_out

    def encoder(self, x_encoder):
        enc_out, state_h, state_c = self.enc_convlstm(x_encoder)
        return enc_out, [state_h, state_c]

    def decoder(self, initial_x_decoder, initial_state=None):
        """decoder follow standard propagation method."""

        h, c = initial_state
        x = initial_x_decoder

        x_stack, h_stack, c_stack = [], [], []
        for i in range(self.dec_len):
            x, [h, c] = self.dec_convlstm_cell(initial_x_decoder=x,
                                               initial_state=[h, c])
            x_stack.append(x)

        return x
