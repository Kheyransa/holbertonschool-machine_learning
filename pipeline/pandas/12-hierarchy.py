#!/usr/bin/env python3
"""
Defines a function that creates a MultiIndex hierarchy with Timestamp on top
"""
import pandas as pd
index_method = __import__('10-index').index


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
    # 1. Hər iki dataframe-i Timestamp-ə görə indeksləyirik
    df1_indexed = index_method(df1)
    df2_indexed = index_method(df2)

    # 2. Birləşdiririk (Öncə bitstamp, sonra coinbase gəlməlidir)
    df = pd.concat([df2_indexed, df1_indexed], keys=['bitstamp', 'coinbase'])

    # 3. İndeks səviyyələrinin yerini dəyişirik (Timestamp önə keçir)
    df = df.swaplevel(0, 1, axis=0)

    # 4. İndeksləri xronoloji olaraq artan sıra ilə çeşidləyirik
    df = df.sort_index()

    # 5. Müəyyən olunmuş zaman aralığını filtrləyirik
    return df.loc[1417411980:1417417980]
