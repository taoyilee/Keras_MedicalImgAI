"""
Modify from DenseNet-Keras (https://github.com/flyyufelix/DenseNet-Keras)
"""
import keras.backend as kb
from keras import regularizers
from keras.layers import Input
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.models import Model

from .custom_layers import Scale


def get_model(class_names, base_weights_path=None, weights_path=None, image_dimension=512, color_mode='grayscale',
              weight_decay=1e-4, class_mode='multiclass'):
    """
    Create model for transfer learning

    Arguments:
    class_names - list of str
    weights_path - str

    Returns:
    model - Keras model
    """

    if weights_path == "":
        weights_path = None

    base_model = densenet121(reduction=0.5, weights_path=base_weights_path, image_dimension=image_dimension,
                             color_mode=color_mode)

    # create our own output
    # x = base_model.get_layer("conv5_blk_scale").output
    x = base_model.output
    # x = GlobalAveragePooling2D()(x)

    # dense layers for different class
    predictions = []
    if class_mode == 'multiclass':
        prediction = Dense(4096, kernel_regularizer=regularizers.l2(weight_decay), name="fc_hidden_layer1")(x)
        predictions = Dense(len(class_names), activation="softmax", name="fc_output_layer",
                            kernel_regularizer=regularizers.l2(weight_decay))(prediction)
    elif class_mode == 'multibinary':
        for i, class_name in enumerate(class_names):
            prediction = Dense(1024, kernel_regularizer=regularizers.l2(weight_decay))(x)
            prediction = Dense(1, kernel_regularizer=regularizers.l2(weight_decay), activation="sigmoid",
                               name=class_name)(prediction)
            predictions.append(prediction)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.base_model = base_model
    if weights_path is not None:
        model.load_weights(weights_path)
    return model


"Modify to Keras 2.0 API from https://github.com/flyyufelix/DenseNet-Keras"


def densenet121(nb_dense_block=4, growth_rate=16, nb_filter=64, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4,
                weights_path=None, image_dimension=512, color_mode='grayscale'):
    """
    Instantiate the DenseNet 121 architecture,
        # Arguments
            nb_dense_block: number of dense blocks to add to end
            growth_rate: number of filters to add per dense block
            nb_filter: initial number of filters
            reduction: reduction factor of transition blocks.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            classes: optional number of classes to classify images
            weights_path: path to pre-trained weights
        # Returns
            A Keras model instance.
    """
    eps = 1.1e-5

    input_channels = {'grayscale': 1, 'rgb': 3, 'hsv': 3}
    # compute compression factor
    compression = 1.0 - reduction

    # Handle Dimension Ordering for different backends
    global concat_axis
    if kb.image_dim_ordering() == 'tf':
        concat_axis = 3
        img_input = Input(shape=(image_dimension, image_dimension, input_channels[color_mode]), name='data')
    else:
        concat_axis = 1
        img_input = Input(shape=(input_channels[color_mode], image_dimension, image_dimension), name='data')

    # From architecture for ImageNet (Table 1 in the paper)
    nb_layers = [6, 12, 24, 16]

    # Initial convolution
    x = Conv2D(nb_filter, (7, 7), strides=(2, 2), padding="same", use_bias=False, name="conv1",
               kernel_regularizer=regularizers.l2(weight_decay))(img_input)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
    x = Scale(axis=concat_axis, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx + 2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter,
                                   growth_rate, dropout_rate=dropout_rate,
                                   weight_decay=weight_decay)

        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate,
                             weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate,
                               weight_decay=weight_decay)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv' + str(final_stage) + '_blk_bn')(x)
    x = Scale(axis=concat_axis, name='conv' + str(final_stage) + '_blk_scale')(x)
    x = Activation('relu', name='relu' + str(final_stage) + '_blk')(x)
    x = GlobalAveragePooling2D(name='pool' + str(final_stage))(x)

    # x = Dense(classes, name='fc6')(x)
    # x = Activation('softmax', name='prob')(x)

    model = Model(img_input, x, name='densenet')

    if weights_path is not None:
        model.load_weights(weights_path)
    return model


def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    """
    Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor 
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    """
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_x1_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base + '_x1_scale')(x)
    x = Activation('relu', name=relu_name_base + '_x1')(x)
    x = Conv2D(inter_channel, (1, 1), use_bias=False, name=f"{conv_name_base}_x1",
               kernel_regularizer=regularizers.l2(weight_decay))(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_x2_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base + '_x2_scale')(x)
    x = Activation('relu', name=relu_name_base + '_x2')(x)
    x = Conv2D(nb_filter, (3, 3), padding="same", use_bias=False, name=f"{conv_name_base}_x2",
               kernel_regularizer=regularizers.l2(weight_decay))(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    """
    Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    """

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base + '_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Conv2D(int(nb_filter * compression), (1, 1), use_bias=False, name=conv_name_base,
               kernel_regularizer=regularizers.l2(weight_decay))(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)
    return x


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4,
                grow_nb_filters=True):
    """
    Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    """
    concat_feat = x
    for i in range(nb_layers):
        branch = i + 1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = concatenate(
            [concat_feat, x],
            axis=concat_axis,
            name='concat_' + str(stage) + '_' + str(branch),
        )
        if grow_nb_filters:
            nb_filter += growth_rate
    return concat_feat, nb_filter
