import re
import numpy as np
from numpy.lib.stride_tricks import as_strided
import pandas as pd


def zero_mean_normalize(df: pd.DataFrame, columns: list):
    """
    Zero mean normalize the given columns in the given dataframe.
    """

    for column in columns:
        df[column] -= df[column].mean()
        df[column] /= df[column].std()


def min_max_normalize(df: pd.DataFrame, columns: list):
    """
    Min-max normalize the given columns in the given dataframe.
    """
    for column in columns:
        df[column] -= df[column].min()
        df[column] /= (df[column].max() - df[column].min())
