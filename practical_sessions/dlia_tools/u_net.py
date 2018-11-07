"""U-Net model implementation with keras"""

import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, GaussianNoise, Dropout


def u_net(shape, nb_filters_0=32, exp=1, conv_size=3, initialization='glorot_uniform', activation="relu", sigma_noise=0, output_channels=1, drop=0.0):
    """U-Net model.

    Standard U-Net model, plus optional gaussian noise.
    Note that the dimensions of the input images should be
    multiples of 16.

    Arguments:
    shape: image shape, in the format (nb_channels, x_size, y_size).
    nb_filters_0 : initial number of filters in the convolutional layer.
    exp : should be equal to 0 or 1. Indicates if the number of layers should be constant (0) or increase exponentially (1).
    conv_size : size of convolution.
    initialization: initialization of the convolutional layers.
    activation: activation of the convolutional layers.
    sigma_noise: standard deviation of the gaussian noise layer. If equal to zero, this layer is deactivated.
    output_channels: number of output channels.
    drop: dropout rate

    Returns:
    U-Net model - it still needs to be compiled.

    Reference:
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    Olaf Ronneberger, Philipp Fischer, Thomas Brox
    MICCAI 2015

    Credits:
    The starting point for the code of this function comes from:
    https://github.com/jocicmarko/ultrasound-nerve-segmentation
    by Marko Jocic
    """

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    inputs = Input(shape)
    conv1 = Conv2D(nb_filters_0, conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv1_1")(inputs)
    conv1 = Conv2D(nb_filters_0, conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv1_2")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    if drop > 0.0: pool1 = Dropout(drop)(pool1)

    conv2 = Conv2D(nb_filters_0 * 2**(1 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv2_1")(pool1)
    conv2 = Conv2D(nb_filters_0 * 2**(1 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv2_2")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    if drop > 0.0: pool2 = Dropout(drop)(pool2)

    conv3 = Conv2D(nb_filters_0 * 2**(2 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv3_1")(pool2)
    conv3 = Conv2D(nb_filters_0 * 2**(2 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv3_2")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    if drop > 0.0: pool3 = Dropout(drop)(pool3)

    conv4 = Conv2D(nb_filters_0 * 2**(3 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv4_1")(pool3)
    conv4 = Conv2D(nb_filters_0 * 2**(3 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv4_2")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    if drop > 0.0: pool4 = Dropout(drop)(pool4)

    conv5 = Conv2D(nb_filters_0 * 2**(4 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv5_1")(pool4)
    conv5 = Conv2D(nb_filters_0 * 2**(4 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv5_2")(conv5)
    if drop > 0.0: conv5 = Dropout(drop)(conv5)

    up6 = concatenate(
        [UpSampling2D(size=(2, 2))(conv5), conv4], axis=channel_axis)
    conv6 = Conv2D(nb_filters_0 * 2**(3 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv6_1")(up6)
    conv6 = Conv2D(nb_filters_0 * 2**(3 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv6_2")(conv6)
    if drop > 0.0: conv6 = Dropout(drop)(conv6)

    up7 = concatenate(
        [UpSampling2D(size=(2, 2))(conv6), conv3], axis=channel_axis)
    conv7 = Conv2D(nb_filters_0 * 2**(2 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv7_1")(up7)
    conv7 = Conv2D(nb_filters_0 * 2**(2 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv7_2")(conv7)
    if drop > 0.0: conv7 = Dropout(drop)(conv7)

    up8 = concatenate(
        [UpSampling2D(size=(2, 2))(conv7), conv2], axis=channel_axis)
    conv8 = Conv2D(nb_filters_0 * 2**(1 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv8_1")(up8)
    conv8 = Conv2D(nb_filters_0 * 2**(1 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv8_2")(conv8)
    if drop > 0.0: conv8 = Dropout(drop)(conv8)

    up9 = concatenate(
        [UpSampling2D(size=(2, 2))(conv8), conv1], axis=channel_axis)
    conv9 = Conv2D(nb_filters_0, conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv9_1")(up9)
    conv9 = Conv2D(nb_filters_0, conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv9_2")(conv9)
    if drop > 0.0: conv9 = Dropout(drop)(conv9)

    if sigma_noise > 0:
        conv9 = GaussianNoise(sigma_noise)(conv9)

    conv10 = Conv2D(output_channels, 1, activation='sigmoid', name="conv_out")(conv9)

    return Model(inputs, conv10)


def u_net3(shape, nb_filters_0=32, exp=1, conv_size=3, initialization='glorot_uniform', activation="relu", sigma_noise=0, output_channels=1):
    """U-Net model, with three layers.

    U-Net model using 3 maxpoolings/upsamplings, plus optional gaussian noise.

    Arguments:
    shape: image shape, in the format (nb_channels, x_size, y_size).
    nb_filters_0 : initial number of filters in the convolutional layer.
    exp : should be equal to 0 or 1. Indicates if the number of layers should be constant (0) or increase exponentially (1).
    conv_size : size of convolution.
    initialization: initialization of the convolutional layers.
    activation: activation of the convolutional layers.
    sigma_noise: standard deviation of the gaussian noise layer. If equal to zero, this layer is deactivated.
    output_channels: number of output channels.

    Returns:
    U-Net model - it still needs to be compiled.

    Reference:
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    Olaf Ronneberger, Philipp Fischer, Thomas Brox
    MICCAI 2015

    Credits:
    The starting point for the code of this function comes from:
    https://github.com/jocicmarko/ultrasound-nerve-segmentation
    by Marko Jocic
    """

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    inputs = Input(shape)
    conv1 = Conv2D(nb_filters_0, conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv1_1")(inputs)
    conv1 = Conv2D(nb_filters_0, conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv1_2")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(nb_filters_0 * 2**(1 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv2_1")(pool1)
    conv2 = Conv2D(nb_filters_0 * 2**(1 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv2_2")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(nb_filters_0 * 2**(2 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv3_1")(pool2)
    conv3 = Conv2D(nb_filters_0 * 2**(2 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv3_2")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(nb_filters_0 * 2**(3 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv4_1")(pool3)
    conv4 = Conv2D(nb_filters_0 * 2**(3 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv4_2")(conv4)

    up5 = concatenate(
        [UpSampling2D(size=(2, 2))(conv4), conv3], axis=channel_axis)
    conv5 = Conv2D(nb_filters_0 * 2**(2 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv5_1")(up5)
    conv5 = Conv2D(nb_filters_0 * 2**(2 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv5_2")(conv5)

    up6 = concatenate(
        [UpSampling2D(size=(2, 2))(conv5), conv2], axis=channel_axis)
    conv6 = Conv2D(nb_filters_0 * 2**(1 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv6_1")(up6)
    conv6 = Conv2D(nb_filters_0 * 2**(1 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv6_2")(conv6)

    up7 = concatenate(
        [UpSampling2D(size=(2, 2))(conv6), conv1], axis=channel_axis)
    conv7 = Conv2D(nb_filters_0, conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv7_1")(up7)
    conv7 = Conv2D(nb_filters_0, conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv7_2")(conv7)

    if sigma_noise > 0:
        conv7 = GaussianNoise(sigma_noise)(conv7)

    conv10 = Conv2D(output_channels, 1, activation='sigmoid', name="conv_out")(conv7)

    return Model(inputs, conv10)
