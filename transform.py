import numpy as np


def one_hot_encoding(y_true: np.ndarray, n_classes) -> np.ndarray:
    """
    One hot encoding of the given array of labels.
    """
    return np.eye(n_classes)[y_true.reshape(-1)]
