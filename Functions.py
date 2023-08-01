import numpy as np


# ------------------------------------------------------------------------------
# DEFINITION OF POSSIBLE COST FUNCTIONS----------------------------------------
# ------------------------------------------------------------------------------

def quadratic_error(prediction, real_output):
    return np.sum(np.power(prediction - real_output, 2)) / len(prediction)


def quadratic_error_derivative(prediction, real_output):
    return prediction - real_output


# ------------------------------------------------------------------------------
# DEFINITION OF POSSIBLE ACTIVATION FUNCTIONS----------------------------------
# ------------------------------------------------------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return np.exp(-x) * np.square(sigmoid(x))
