"""
Modify from DenseNet-Keras (https://github.com/flyyufelix/DenseNet-Keras)
"""
from keras import regularizers
from keras.applications.densenet import DenseNet121
from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Dense
from keras.models import Model


def get_model(class_names, base_weights=None, weights_path=None, image_dimension=224, color_mode='grayscale',
              weight_decay=1e-4, class_mode='multiclass', final_activation="softmax"):
    """
    Create model for transfer learning

    Arguments:
    class_names - list of str
    weights_path - str

    Returns:
    model - Keras model
    """
    if weights_path is not None:
        base_weights = None
    base_model = DenseNet121(include_top=False, weights=base_weights, pooling="None")
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # dense layers for different class
    predictions = []
    if class_mode == 'multiclass':
        prediction = Dense(4096, kernel_regularizer=regularizers.l2(weight_decay), name="fc_hidden_layer1")(x)
        predictions = Dense(len(class_names), activation=final_activation, name="fc_output_layer",
                            kernel_regularizer=regularizers.l2(weight_decay))(prediction)
    elif class_mode == 'multibinary':
        for i, class_name in enumerate(class_names):
            prediction = Dense(1024, kernel_regularizer=regularizers.l2(weight_decay))(x)
            prediction = Dense(1, kernel_regularizer=regularizers.l2(weight_decay), activation="sigmoid",
                               name=class_name)(prediction)
            predictions.append(prediction)

    model = Model(inputs=base_model.input, outputs=predictions)
    if weights_path is not None:
        model.load_weights(weights_path)
    model.base_model = base_model
    return model
