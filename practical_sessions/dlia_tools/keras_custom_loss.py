from __future__ import print_function

import numpy as np
import tensorflow
from tensorflow.python.keras import backend as K

# import pdb

SMOOTH = 1.0

#  dice_coef and dice_coef_loss have been borrowed from:
#  https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py


def dice1_coef(y_true, y_pred, smooth=SMOOTH):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice1_loss(y_true, y_pred, smooth=SMOOTH):
    return 1 - dice1_coef(y_true, y_pred, smooth)


def dice2_coef(y_true, y_pred, smooth=SMOOTH):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)


def dice2_loss(y_true, y_pred, smooth=SMOOTH):
    return 1 - dice2_coef(y_true, y_pred, smooth)


def jaccard2_coef(y_true, y_pred, smooth=SMOOTH):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def jaccard2_loss(y_true, y_pred, smooth=SMOOTH):
    return 1 - jaccard2_coef(y_true, y_pred, smooth)


def jaccard1_coef(y_true, y_pred, smooth=SMOOTH):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def jaccard1_loss(y_true, y_pred, smooth=SMOOTH):
    return 1 - jaccard1_coef(y_true, y_pred, smooth=SMOOTH)


def diag_dist_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return K.mean(0.5 - 0.5 * np.cos(np.pi * np.abs(y_true_f - y_pred_f)))


def cross_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    # return K.mean(y_true_f + y_pred_f -2*y_true_f*y_pred_f)
    return K.mean((y_true_f - y_pred_f)**4)


def rmse(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    # return K.mean(y_true_f + y_pred_f -2*y_true_f*y_pred_f)
    return K.sqrt(K.mean(K.square(y_true_f - y_pred_f)))
