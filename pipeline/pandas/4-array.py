#!/usr/bin/env python3
"""
Defines a function that converts specific DataFrame rows to a numpy array
"""


def array(df):
    """
    Takes a pd.DataFrame, selects the last 10 rows of High and Close columns,
    and converts them into a numpy.ndarray.

    Parameters:
        df: The input DataFrame

    Returns:
        numpy.ndarray: The selected data converted to a numpy array
    """
    return df[['High', 'Close']].tail(10).to_numpy()
