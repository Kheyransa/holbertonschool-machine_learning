#!/usr/bin/env python3
"""
Defines a function that renames a column and modifies values
"""
import pandas as pd


def rename(df):
    """
    Renames Timestamp column to Datetime, converts it to datetime format,
    and returns only Datetime and Close columns.

    Parameters:
        df (pd.DataFrame): The input DataFrame

    Returns:
        pd.DataFrame: The modified DataFrame
    """
    df = df.rename(columns={'Timestamp': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    return df[['Datetime', 'Close']]
