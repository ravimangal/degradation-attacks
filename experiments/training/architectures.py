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

from gloro.layers import InvertibleDownsampling as InvertibleDownSampling
from gloro.layers import MinMax
from gloro.layers import ResnetBlock


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