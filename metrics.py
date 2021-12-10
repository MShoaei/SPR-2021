import numpy as np


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred)**2)


def cross_entropy_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # assert y_true.shape == y_pred.shape and (len(y_true.shape) == 1 or y_true.shape[0] == 1), \
    #     "y_true and y_pred must be 1D arrays of the same length"
    return -np.mean(y_true * np.log(y_pred))


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(y_true == y_pred)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return 2 * (y_true * y_pred).sum() / (y_true + y_pred).sum()


def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return (y_true * y_pred).sum() / y_pred.sum()


def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return (y_true * y_pred).sum() / y_true.sum()
