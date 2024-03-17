import tensorflow as tf
from tensorflow.keras import backend as K
from keras import backend as keras


# metrics
def dice_coefficient(y_true, y_pred):
    smooth = 1e-6  # 为了避免除以零
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def iou(y_true, y_pred):
    smooth = 1e-6
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=[1, 2, 3])
    iou = (intersection + smooth) / (sum_ - intersection + smooth)
    return iou


def dice_loss(y_true, y_pred):
    """
    Dice loss, a measure of overlap between two samples.
    """
    smooth = 1.0  # Smooth factor to avoid division by zero
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + smooth)
