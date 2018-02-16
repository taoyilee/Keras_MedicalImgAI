import cv2
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D
from keras.models import Model

from app.main.Actions import Actions
from app.models.model_factory import get_model


class Visualizer(Actions):
    y = None
    y_hat = None
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    model: Model = []
    test_generator = None

    def __init__(self, config_file, weight):
        super().__init__(config_file)
        if weight is not None:
            self.weight = weight
        else:
            self.MDConfig.use_trained_model_weights = True

            self.weight = self.MDConfig.trained_model_weights
        print(f"** Visualize weight file: {self.weight}")

    def prepare_model(self):
        print("** load model **")
        self.MDConfig.use_trained_model_weights = True
        print(f"** Trained Model = {self.MDConfig.trained_model_weights} **")
        self.model = get_model(self.DSConfig.class_names, weights_path=self.MDConfig.trained_model_weights,
                               image_dimension=self.IMConfig.img_dim, color_mode=self.IMConfig.color_mode,
                               class_mode=self.DSConfig.class_mode)

    def kernel_visualize(self):
        self.prepare_model()
        if self.MDConfig.show_model_summary:
            self.model.summary()
        layer: Conv2D = self.model.get_layer("conv1/conv")

        weights: tf.Variable = layer.weights[0]
        weights = np.array(K.eval(weights))
        weights -= np.min(weights)
        weights /= np.max(weights)
        weights *= 255
        weights_mosaic = np.zeros((56, 56, 3))
        for i in range(8):
            for j in range(8):
                weights_mosaic[i * 7:(i + 1) * 7, j * 7:(j + 1) * 7, :] = weights[:, :, :, i * 8 + j].squeeze()
        weights_mosaic = cv2.resize(weights_mosaic, (1024, 1024))
        cv2.imwrite("kernels.png", weights_mosaic)
