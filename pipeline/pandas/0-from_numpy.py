#!/usr/bin/env python3
import pandas as pd


def from_numpy(array):
    num_cols = array.shape[1]
    column_names = [chr(65 + i) for i in range(num_cols)]
    return pd.DataFrame(array, columns=column_names)


from_numpy.__doc__ = "Creates a pd.DataFrame from a np.ndarray"
import sys
sys.modules[__name__].__doc__ = "Defines from_numpy function"
