#!/usr/bin/env python3
"""
Defines a function that concatenates two pd.DataFrames
"""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
    Indexes both dataframes on Timestamp, filters df2 up to 1417411920,
    and concatenates df2 on top of df1 with custom keys.

    Parameters:
        df1: Coinbase DataFrame
        df2: Bitstamp DataFrame

    Returns:
        pd.DataFrame: The concatenated DataFrame
    """
    df1_indexed = index(df1)
    df2_indexed = index(df2)

    df2_filtered = df2_indexed.loc[:1417411920]

    return pd.concat([df2_filtered, df1_indexed], keys=['bitstamp', 'coinbase'])
