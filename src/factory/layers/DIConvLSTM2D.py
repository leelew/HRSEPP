import tensorflow as tf
from tensorflow.keras import Input, Model, layers
from tensorflow.python.keras.layers.convolutional_recurrent import \
    ConvLSTM2DCell


class DIConvLSTM(layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='same'):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        
        super().__init__()

    def build(self, input_shape):
        self.convlstm = []
        self.t = input_shape[1]

        for i in range(self.t):
            self.convlstm.append(ConvLSTM2DCell(filters=self.filters, 
                                                kernel_size=self.kernel_size, 
                                                strides=self.strides, 
                                                padding=self.padding))
        self.dense = layers.Dense(1)

        
        self.built = True

    def call(self, inputs):
        # must inputs[:,0] don't have nan

        h0 = tf.zeros_like(inputs[:,0])
        h = tf.tile(h0, [1, 1, 1, self.filters])
        c = tf.tile(h0, [1, 1, 1, self.filters])
        print(h.shape)
        print(c.shape)

        h_all = []
        for i in range(self.t):   
            print(i)
            x = inputs[:, i]
            print(x.shape)
            if i > 0:   
                #FIXME: Don't know how to replace NaN with predict 
                #       value in tensorflow. Thus use mean value for 
                #       obs and predict images.
                m = tf.stack([x, out], axis=0)
                x = tf.experimental.numpy.nanmean(m, axis=0)
                #mask = tf.where(tf.math.is_nan(x))
                #mask = x == x
                #print(mask)
                #x = tf.tensor_scatter_nd_update(x, mask, out[mask])
                #x[mask] = a[mask]

            out, [h, c] = self.convlstm[i](x, [h, c])
            print(out.shape)
            out = self.dense(out)
            print(out.shape)
            h_all.append(h)
        
        return tf.stack(h_all, axis=1)


if __name__ == '__main__':

    x = Input((7, 112, 112, 1))
    out = DIConvLSTM(16, 3)(x)
    mdl = Model(x, out)
    mdl.summary()
