#!/usr/bin/env python3
import pandas as pd
import string


def from_numpy(array):
    num_cols = array.shape[1]
    column_names = list(string.ascii_uppercase[:num_cols])
    return pd.DataFrame(array, columns=column_names)
