#!/usr/bin/env python3
"""
Defines a function that fills missing values in a pd.DataFrame
"""


def fill(df):
    """
    Cleans and fills missing values in the cryptocurrency dataframe.

    Parameters:
        df: The input DataFrame

    Returns:
        pd.DataFrame: The modified DataFrame
    """
    df = df.drop(columns=['Weighted_Price'])

    df['Close'] = df['Close'].ffill()

    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Open'] = df['Open'].fillna(df['Close'])

    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = (
        df['Volume_(Currency)'].fillna(0)
    )

    return df
