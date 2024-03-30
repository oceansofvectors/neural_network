import numpy as np


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def categorical_cross_entropy(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
