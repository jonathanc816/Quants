import numpy as np


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))