#!/usr/bin/env python3
"""
Defines a function that prunes a pd.DataFrame by removing NaN values
"""


def prune(df):
    """
    Removes any entries where the Close column has NaN values.

    Parameters:
        df: The input DataFrame

    Returns:
        pd.DataFrame: The modified DataFrame
    """
    return df.dropna(subset=['Close'])
