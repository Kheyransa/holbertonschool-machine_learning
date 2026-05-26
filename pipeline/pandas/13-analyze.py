#!/usr/bin/env python3
"""
Defines a function that computes descriptive statistics for a pd.DataFrame
"""


def analyze(df):
    """
    Computes descriptive statistics for all columns except Timestamp.

    Parameters:
        df: The input DataFrame

    Returns:
        pd.DataFrame: A new DataFrame containing the statistics
    """
    return df.drop(columns=['Timestamp']).describe()
