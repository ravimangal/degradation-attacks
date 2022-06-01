import tensorflow.keras.backend as K
import tensorflow as tf

#from deel.lip.initializers import BjorckInitializer
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization

from gloro.v1.layers import InvertibleDownsampling as InvertibleDownSampling
from gloro.v1.layers import MinMax
from gloro.v1.layers import ResnetBlock


def _get_initializer(initialization):
    if initialization == 'orthogonal':
        return Orthogonal()

    elif initialization == 'glorot_uniform':
        return GlorotUniform()

    # elif initialization == 'bjorck':
    #     return BjorckInitializer()

    else:
        raise ValueError(f'unknown initialization: {initialization}')

def _add_pool(z, pooling_type, activation=None, initializer=None):
    if pooling_type == 'avg':
        return AveragePooling2D()(z)

    elif pooling_type == 'conv':
        channels = K.int_shape(z)[-1]

        if initializer is None:
            initializer = _get_initializer('orthogonal')

        z = Conv2D(
            channels, 
            4, 
            strides=2, 
            padding='same', 
            kernel_initializer=initializer)(z)

        return _add_activation(z, activation)

    elif pooling_type == 'invertible':
        return InvertibleDownSampling()(z)

    else:
        raise ValueError(f'unknown pooling type: {pooling_type}')

def _add_activation(z, activation_type='relu'):
    if activation_type == 'relu':
        return Activation('relu')(z)

    elif activation_type == 'elu':
        return Activation('elu')(z)

    elif activation_type == 'softplus':
        return Activation('softplus')(z)

    elif activation_type == 'minmax':
        return MinMax()(z)

    else:
        raise ValueError(f'unknown activation type: {activation_type}')


def acas_dense(
        input_shape, 
        num_classes, 
        activation='relu',
        initialization='orthogonal'):

    initializer = _get_initializer(initialization)

    x = Input(input_shape)

    z = x
    for _ in range(6):
        z = Dense(50, kernel_initializer=initializer)(z)
        z = _add_activation(z, activation)

    y = Dense(num_classes, kernel_initializer=initializer)(z)

    return x, y

def minmax_acas_dense(input_shape, num_classes, initialization='orthogonal'):

    return acas_dense(
        input_shape, num_classes, 
        activation='minmax', 
        initialization=initialization)

def acas_dense_large(
        input_shape, 
        num_classes, 
        activation='relu',
        initialization='orthogonal'):

    initializer = _get_initializer(initialization)

    x = Input(input_shape)

    z = x
    for _ in range(3):
        z = Dense(1000, kernel_initializer=initializer)(z)
        z = _add_activation(z, activation)

    y = Dense(num_classes, kernel_initializer=initializer)(z)

    return x, y

def minmax_acas_dense_large(
    input_shape, num_classes, initialization='orthogonal'):

    return acas_dense_large(
        input_shape, num_classes, 
        activation='minmax', 
        initialization=initialization)


def cnn_2C2F(
        input_shape, 
        num_classes,
        pooling='conv',
        activation='relu',
        initialization='orthogonal'):
    
    initializer = _get_initializer(initialization)

    x = Input(input_shape)
    z = Conv2D(
        16, 4, strides=2, padding='same', kernel_initializer=initializer)(x)
    z = _add_activation(z, activation)

    z = Conv2D(
        32, 4, strides=2, padding='same', kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)

    z = Flatten()(z)
    z = Dense(100, kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    y = Dense(num_classes, kernel_initializer=initializer)(z)

    return x, y


def minmax_cnn_2C2F(
        input_shape,
        num_classes,
        pooling='conv',
        initialization='orthogonal'):

    return cnn_2C2F(
        input_shape, num_classes,
        pooling=pooling,
        activation='minmax',
        initialization=initialization)


def cnn_4C3F(
        input_shape, 
        num_classes, 
        pooling='conv', 
        activation='relu',
        initialization='orthogonal'):
    
    initializer = _get_initializer(initialization)

    x = Input(input_shape)
    z = Conv2D(32, 3, padding='same', kernel_initializer=initializer)(x)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initializer)

    z = Conv2D(64, 3, padding='same', kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initializer)

    z = Flatten()(z)
    z = Dense(512, kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    z = Dense(512, kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    y = Dense(num_classes, kernel_initializer=initializer)(z)

    return x, y


def minmax_cnn_4C3F(
        input_shape,
        num_classes,
        pooling='conv',
        initialization='orthogonal'):

    return cnn_4C3F(
        input_shape, num_classes,
        pooling=pooling,
        activation='minmax',
        initialization=initialization)


def cnn_6C2F(
        input_shape, 
        num_classes, 
        pooling='conv', 
        activation='relu',
        initialization='orthogonal'):
    
    initializer = _get_initializer(initialization)

    x = Input(input_shape)
    z = Conv2D(32, 3, padding='same', kernel_initializer=initializer)(x)
    z = _add_activation(z, activation)
    z = Conv2D(32, 3, padding='same', kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initializer)

    z = Conv2D(64, 3, padding='same', kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    z = Conv2D(64, 3, padding='same', kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initializer)

    z = Flatten()(z)
    z = Dense(512, kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    y = Dense(num_classes, kernel_initializer=initializer)(z)

    return x, y


def minmax_cnn_6C2F(
        input_shape, 
        num_classes, 
        pooling='conv', 
        initialization='orthogonal'):

    return cnn_6C2F(
        input_shape, num_classes, 
        pooling=pooling, 
        activation='minmax',
        initialization=initialization)


def cnn_8C2F(
        input_shape, 
        num_classes, 
        pooling='conv', 
        activation='relu',
        initialization='orthogonal'):
    
    initializer = _get_initializer(initialization)

    x = Input(input_shape)
    z = Conv2D(64, 3, padding='same', kernel_initializer=initializer)(x)
    z = _add_activation(z, activation)
    z = Conv2D(64, 3, padding='same', kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initializer)

    z = Conv2D(128, 3, padding='same', kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    z = Conv2D(128, 3, padding='same', kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initializer)

    z = Conv2D(256, 3, padding='same', kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initializer)

    z = Flatten()(z)
    z = Dense(256, kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    y = Dense(num_classes, kernel_initializer=initializer)(z)

    return x, y


def minmax_cnn_8C2F(
        input_shape, 
        num_classes, 
        pooling='conv', 
        initialization='orthogonal'):

    return cnn_8C2F(
        input_shape, num_classes, 
        pooling=pooling, 
        activation='minmax',
        initialization=initialization)


def alexnet(
        input_shape,
        num_classes,
        pooling='avg',
        activation='relu',
        initialization='orthogonal',
        dropout=False):

    initializer = _get_initializer(initialization)

    x = Input(input_shape)

    z = Conv2D(
        96,
        11,
        padding='same',
        strides=4,
        kernel_initializer=initializer)(x)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initializer)

    z = Conv2D(256, 5, padding='same', kernel_initializer=initializer)(z)
    z = Activation('relu')(z)
    z = _add_pool(z, pooling, activation, initializer)

    z = Conv2D(384, 3, padding='same', kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    z = Conv2D(384, 3, padding='same', kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    z = Conv2D(384, 3, padding='same', kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initializer)

    z = Flatten()(z)
    if dropout:
        z = Dropout(0.5)(z)
    z = Dense(4096, kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    if dropout:
        z = Dropout(0.5)(z)
    z = Dense(4096, kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    
    y = Dense(num_classes, kernel_initializer=initializer)(z)

    return x, y

def minmax_alexnet(
        input_shape,
        num_classes,
        pooling='invertible',
        initialization='orthogonal',
        dropout=False):

    return alexnet(
        input_shape, num_classes, 
        pooling=pooling,
        activation='minmax',
        initialization=initialization,
        dropout=dropout)


def vgg16(
        input_shape,
        num_classes,
        pooling='avg',
        activation='relu',
        initialization='orthogonal'):

    initializer = _get_initializer(initialization)

    x = Input(input_shape)

    z = Conv2D(64, 3, padding='same', kernel_initializer=initializer)(x)
    z = _add_activation(z, activation)
    z = Conv2D(64, 3, padding='same', kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initializer)

    z = Conv2D(128, 3, padding='same', kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    z = Conv2D(128, 3, padding='same', kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initializer)

    z = Conv2D(256, 3, padding='same', kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    z = Conv2D(256, 3, padding='same', kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    z = Conv2D(256, 3, padding='same', kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initializer)

    z = Conv2D(512, 3, padding='same', kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    z = Conv2D(512, 3, padding='same', kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    z = Conv2D(512, 3, padding='same', kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initializer)

    z = Conv2D(512, 3, padding='same', kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    z = Conv2D(512, 3, padding='same', kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    z = Conv2D(512, 3, padding='same', kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initializer)

    z = Flatten()(z)
    z = Dense(4096, kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    z = Dense(4096, kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)

    y = Dense(num_classes, kernel_initializer=initializer)(z)

    return x, y

def minmax_vgg16(
        input_shape,
        num_classes,
        pooling='invertible',
        initialization='orthogonal'):

    return vgg16(
        input_shape, num_classes, 
        pooling=pooling,
        activation='minmax',
        initialization=initialization)


def resnet50(
        # input=None,
             input_shape=None,
             use_ortho_init=True,
             identity_skip=False,
             use_batchnorm=False,
             use_invertible_downsample=True,
             use_weighted_merge=False,
             num_classes=1000):
    # if input is None and input_shape is None:
    #     raise ValueError(
    #         "One of the input layer or the shape of the input layer must be not be `None`."
    #     )

    # if input is None:
    input = Input(input_shape)

    if use_ortho_init:
        dense_init = tf.keras.initializers.Orthogonal
    else:
        dense_init = tf.keras.initializers.GlorotUniform

    z = input

    z = Conv2D(64, (7, 7), strides=(2, 2))(z)
    z = Activation('relu')(z)

    if use_batchnorm:
        z = BatchNormalization(
            gamma_initializer=tf.keras.initializers.Constant(0.5),
            trainable=False)(z)

    z = Conv2D(64, (3, 3), strides=(2, 2))(z)  # replace maxpool
    z = Activation('relu')(z)

    if use_batchnorm:
        z = BatchNormalization(
            gamma_initializer=tf.keras.initializers.Constant(0.5),
            trainable=False)(z)

    # replace max-pool with average pool
    z = AveragePooling2D()(z)

    for _ in range(3):
        z = ResnetBlock(filters=(64, 64, 256),
                        kernel_sizes=(1, 3, 1),
                        stride1=1,
                        use_ortho_init=use_ortho_init,
                        identity_skip=identity_skip,
                        use_batchnorm=use_batchnorm,
                        use_invertible_downsample=True,
                        use_weighted_merge=use_weighted_merge)(z)
    for _ in range(4):
        z = ResnetBlock(filters=(128, 128, 512),
                        kernel_sizes=(1, 3, 1),
                        stride1=1,
                        use_ortho_init=use_ortho_init,
                        identity_skip=identity_skip,
                        use_batchnorm=use_batchnorm,
                        use_invertible_downsample=True,
                        use_weighted_merge=use_weighted_merge
                        )(z)
    for _ in range(6):
        z = ResnetBlock(filters=(256, 256, 1024),
                        kernel_sizes=(1, 3, 1),
                        stride1=1,
                        use_ortho_init=use_ortho_init,
                        identity_skip=identity_skip,
                        use_batchnorm=use_batchnorm,
                        use_invertible_downsample=True,
                        use_weighted_merge=use_weighted_merge)(z)
    for _ in range(3):
        z = ResnetBlock(filters=(512, 512, 2048),
                        kernel_sizes=(1, 3, 1),
                        stride1=2,
                        use_ortho_init=use_ortho_init,
                        identity_skip=identity_skip,
                        use_batchnorm=use_batchnorm,
                        use_invertible_downsample=True,
                        use_weighted_merge=use_weighted_merge)(z)
    z = AveragePooling2D()(z)
    z = Flatten()(z)
    z = Dense(1000, kernel_initializer=dense_init())(z)
    z = Activation('relu')(z)

    y = Dense(num_classes, kernel_initializer=dense_init())(z)

    return input, y


def minmax_resnet50(
        # input=None,
                    input_shape=None,
                    use_ortho_init=True,
                    identity_skip=False,
                    use_batchnorm=True,
                    use_invertible_downsample=True,
                    use_weighted_merge=False,
                    num_classes=1000):
    # if input is None and input_shape is None:
    #     raise ValueError(
    #         "One of the input layer or the shape of the input layer must be not be `None`."
    #     )

    # if input is None:
    input = Input(input_shape)

    if use_ortho_init:
        dense_init = tf.keras.initializers.Orthogonal
    else:
        dense_init = tf.keras.initializers.GlorotUniform

    z = input

    z = Conv2D(64, (7, 7), strides=(1, 1))(z)
    z = InvertibleDownSampling((2, 2))(z)
    z = MinMax()(z)

    if use_batchnorm:
        z = BatchNormalization(
            gamma_initializer=tf.keras.initializers.Constant(0.5),
            trainable=False)(z)

    z = Conv2D(64, (3, 3), strides=(1, 1))(z)  # replace maxpool
    z = InvertibleDownSampling((2, 2))(z)
    z = MinMax()(z)

    if use_batchnorm:
        z = BatchNormalization(
            gamma_initializer=tf.keras.initializers.Constant(0.5),
            trainable=False)(z)

    # replace max-pool with average pool
    z = AveragePooling2D()(z)

    for _ in range(3):
        z = ResnetBlock(filters=(64, 64, 256),
                        kernel_sizes=(1, 3, 1),
                        stride1=1,
                        use_ortho_init=use_ortho_init,
                        identity_skip=identity_skip,
                        use_batchnorm=use_batchnorm,
                        use_invertible_downsample=use_invertible_downsample,
                        activation=MinMax,
                        use_weighted_merge=use_weighted_merge)(z)
    for _ in range(4):
        z = ResnetBlock(filters=(128, 128, 512),
                        kernel_sizes=(1, 3, 1),
                        stride1=1,
                        use_ortho_init=use_ortho_init,
                        identity_skip=identity_skip,
                        use_batchnorm=use_batchnorm,
                        use_invertible_downsample=use_invertible_downsample,
                        activation=MinMax,
                        use_weighted_merge=use_weighted_merge
                        )(z)
    for _ in range(6):
        z = ResnetBlock(filters=(256, 256, 1024),
                        kernel_sizes=(1, 3, 1),
                        stride1=1,
                        use_ortho_init=use_ortho_init,
                        identity_skip=identity_skip,
                        use_batchnorm=use_batchnorm,
                        use_invertible_downsample=use_invertible_downsample,
                        activation=MinMax,
                        use_weighted_merge=use_weighted_merge)(z)
    for _ in range(3):
        z = ResnetBlock(filters=(512, 512, 2048),
                        kernel_sizes=(1, 3, 1),
                        stride1=2,
                        use_ortho_init=use_ortho_init,
                        identity_skip=identity_skip,
                        use_batchnorm=use_batchnorm,
                        use_invertible_downsample=use_invertible_downsample,
                        activation=MinMax,
                        use_weighted_merge=use_weighted_merge)(z)
    z = AveragePooling2D()(z)
    z = Flatten()(z)
    z = Dense(1000, kernel_initializer=dense_init())(z)
    z = MinMax()(z)

    y = Dense(num_classes, kernel_initializer=dense_init())(z)

    return input, y


def resnet50_cifar(
                    # input=None,
                   input_shape=None,
                   use_ortho_init=True,
                   identity_skip=False,
                   use_batchnorm=True,
                   num_classes=1000):
    # if input is None and input_shape is None:
    #     raise ValueError(
    #         "One of the input layer or the shape of the input layer must be not be `None`."
    #     )

    # if input is None:
    input = Input(input_shape)

    if use_ortho_init:
        dense_init = tf.keras.initializers.Orthogonal
    else:
        dense_init = tf.keras.initializers.GlorotUniform

    z = input

    z = Conv2D(64, (3, 3), strides=(1, 1))(z)
    z = Activation('relu')(z)

    if use_batchnorm:
        z = BatchNormalization(
            gamma_initializer=tf.keras.initializers.Constant(0.5),
            trainable=False)(z)

    for l in range(3):
        z = ResnetBlock(filters=(64, 64, 256),
                        kernel_sizes=(1, 3, 1),
                        stride1=1 if l > 0 else 2,
                        use_ortho_init=use_ortho_init,
                        identity_skip=identity_skip,
                        use_batchnorm=use_batchnorm)(z)
    for l in range(4):
        z = ResnetBlock(filters=(128, 128, 512),
                        kernel_sizes=(1, 3, 1),
                        stride1=1 if l > 0 else 2,
                        use_ortho_init=use_ortho_init,
                        identity_skip=identity_skip,
                        use_batchnorm=use_batchnorm)(z)
    for l in range(6):
        z = ResnetBlock(filters=(256, 256, 1024),
                        kernel_sizes=(1, 3, 1),
                        stride1=1 if l > 0 else 2,
                        use_ortho_init=use_ortho_init,
                        identity_skip=identity_skip,
                        use_batchnorm=use_batchnorm)(z)
    for l in range(3):
        z = ResnetBlock(filters=(512, 512, 2048),
                        kernel_sizes=(1, 3, 1),
                        stride1=1 if l > 0 else 2,
                        use_ortho_init=use_ortho_init,
                        identity_skip=identity_skip,
                        use_batchnorm=use_batchnorm)(z)
    z = AveragePooling2D()(z)
    z = Flatten()(z)
    z = Dense(1000, kernel_initializer=dense_init())(z)
    z = Activation('relu')(z)

    y = Dense(num_classes, kernel_initializer=dense_init())(z)

    return input, y


def resnet101(
            # input=None,
              input_shape=None,
              use_ortho_init=True,
              identity_skip=False,
              use_batchnorm=False,
              num_classes=1000):
    # if input is None and input_shape is None:
    #     raise ValueError(
    #         "One of the input layer or the shape of the input layer must be not be `None`."
    #     )

    # if input is None:
    input = Input(input_shape)

    z = input

    z = Conv2D(64, (7, 7), strides=(2, 2))(z)
    z = Activation('relu')(z)
    z = Conv2D(64, (3, 3), strides=(2, 2))(z)  # replace maxpool
    z = Activation('relu')(z)
    for _ in range(3):
        z = ResnetBlock(filters=(64, 64, 256),
                        kernel_sizes=(1, 3, 1),
                        stride1=1,
                        use_ortho_init=use_ortho_init,
                        identity_skip=identity_skip,
                        use_batchnorm=use_batchnorm)(z)
    for _ in range(4):
        z = ResnetBlock(filters=(128, 128, 512),
                        kernel_sizes=(1, 3, 1),
                        stride1=1,
                        use_ortho_init=use_ortho_init,
                        identity_skip=identity_skip,
                        use_batchnorm=use_batchnorm)(z)
    for _ in range(23):
        z = ResnetBlock(filters=(256, 256, 1024),
                        kernel_sizes=(1, 3, 1),
                        stride1=1,
                        use_ortho_init=use_ortho_init,
                        identity_skip=identity_skip,
                        use_batchnorm=use_batchnorm)(z)
    for _ in range(3):
        z = ResnetBlock(filters=(512, 512, 2048),
                        kernel_sizes=(1, 3, 1),
                        stride1=2,
                        use_ortho_init=use_ortho_init,
                        identity_skip=identity_skip,
                        use_batchnorm=use_batchnorm)(z)
    z = AveragePooling2D()(z)
    z = Flatten()(z)
    z = Dense(1000)(z)
    z = Activation('relu')(z)

    y = Dense(num_classes)(z)

    return input, y
