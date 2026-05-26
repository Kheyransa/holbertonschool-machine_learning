#!/usr/bin/env python3
"""
Defines a function that creates a MultiIndex hierarchy with Timestamp on top
"""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Concatenates two dataframes, swaps index levels so Timestamp is level 0,
    sorts chronologically, and slices a specific timestamp range.

    Parameters:
        df1: Coinbase DataFrame
        df2: Bitstamp DataFrame

    Returns:
        pd.DataFrame: The rearranged and sliced MultiIndex DataFrame
    """
    df1_indexed = index(df1)
    df2_indexed = index(df2)

    df = pd.concat([df2_indexed, df1_indexed], keys=['bitstamp', 'coinbase'])
    df = df.swaplevel(0, 1, axis=0)
    df = df.sort_index()

    return df.loc[1417411980:1417417980]
