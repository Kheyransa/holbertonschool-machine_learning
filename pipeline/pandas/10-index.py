#!/usr/bin/env python3
"""
Defines a function that sets a specific column as the index of a pd.DataFrame
"""


def index(df):
    """
    Sets the Timestamp column as the index of the dataframe.

    Parameters:
        df: The input DataFrame

    Returns:
        pd.DataFrame: The modified DataFrame with Timestamp as index
    """
    return df.set_index('Timestamp')
