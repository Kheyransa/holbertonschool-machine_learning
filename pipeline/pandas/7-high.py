#!/usr/bin/env python3
"""
Defines a function that sorts a pd.DataFrame by a specific column
"""


def high(df):
    """
    Sorts the dataframe by the High price column in descending order.

    Parameters:
        df: The input DataFrame

    Returns:
        pd.DataFrame: The sorted DataFrame
    """
    return df.sort_values(by='High', ascending=False)
