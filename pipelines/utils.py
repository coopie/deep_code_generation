import numpy as np


def reshape_for_cnn(example):
    """
    Takes an exmaple of shape (a, b) and returns an example of shape (1, a, b)
    """
    return np.reshape(example, (*example.shape, 1))
