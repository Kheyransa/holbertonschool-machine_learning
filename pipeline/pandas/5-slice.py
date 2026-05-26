#!/usr/bin/env python3
"""
Defines a function that slices a pd.DataFrame
"""


def slice(df):
    """
    Extracts High, Low, Close, and Volume_(BTC) columns
    and selects every 60th row.

    Parameters:
        df: The input DataFrame

    Returns:
        pd.DataFrame: The sliced DataFrame
    """
    return df[['High', 'Low', 'Close', 'Volume_(BTC)']][::60]
