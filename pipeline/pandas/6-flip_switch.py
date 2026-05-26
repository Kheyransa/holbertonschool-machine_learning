#!/usr/bin/env python3
"""
Defines a function that sorts and transposes a pd.DataFrame
"""


def flip_switch(df):
    """
    Sorts the dataframe in reverse chronological order based on Timestamp,
    and transposes the result.

    Parameters:
        df: The input DataFrame

    Returns:
        pd.DataFrame: The transformed DataFrame
    """
    return df.sort_values(by='Timestamp', ascending=False).T
