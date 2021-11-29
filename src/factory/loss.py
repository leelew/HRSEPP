import tensorflow as tf
from tensorflow.keras import losses


class MaskSeq2seqLoss(losses.Loss):
    def __init__(self, mask):
        super().__init__()
        self.mask = mask

    def call(self, y_true, y_pred):
        pass

class MaskMSELoss(tf.keras.losses.Loss):
    def __init__(self, mask):
        super().__init__()
        self.mask = mask

    def call(self, y_true, y_pred):
        mse = tf.math.reduce_mean(tf.square(y_true - y_pred), axis=0)
        mask_mse = tf.math.multiply(mse, self.mask)
        return tf.math.reduce_mean(mask_mse)


class MaskSSIMLoss(tf.keras.losses.Loss):
    def __init__(self, mask):
        super().__init__()
        self.mask = mask

    def call(self, y_true, y_pred):
        y_true_ = tf.math.multiply(y_true, self.mask)
        y_pred_ = tf.math.multiply(y_pred, self.mask)

        return 1 - tf.reduce_mean(
            tf.image.ssim(y_true_, y_pred_, max_val=1.0, filter_size=3))
