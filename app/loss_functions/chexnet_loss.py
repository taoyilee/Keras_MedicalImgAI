import keras.backend as K
import numpy as np
import tensorflow as tf


class chexnet_loss:
    wp = np.ones(14, dtype=K.floatx())
    wn = np.ones(14, dtype=K.floatx())
    eps = 1e-15

    def __init__(self, class_names, class_labels):
        self.class_names = class_names
        self.class_labels = np.array(class_labels)
        self.total_count = np.shape(class_labels)[0]
        print(f"** Total images = {self.total_count}**")
        print(f"** Compute class weights from training data from {np.shape(self.class_labels)}**")
        self.class_weights()

    def class_weights(self):
        for i in range(len(self.class_names)):
            pos_count = np.sum(self.class_labels[..., i], dtype=K.floatx())
            self.wn[i] = pos_count / self.total_count
            self.wp[i] = 1 - self.wn[i]
            print(" [{:>2}] {:>18} W+: {:.3f} W-: {:.3f} W+/W-: {:>7.3f}".format(i + 1, self.class_names[i], self.wp[i],
                                                                                 self.wn[i], self.wp[i] / self.wn[i]))

    def loss_function(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, self.eps, 1 - self.eps)
        cross_entropy_p = tf.multiply(tf.multiply(self.wp, y_true), tf.log(y_pred))
        cross_entropy_n = tf.multiply(tf.multiply(self.wn, 1 - y_true), tf.log(1 - y_pred))
        cross_entropy = -(cross_entropy_p + cross_entropy_n)
        return tf.reduce_sum(cross_entropy)
