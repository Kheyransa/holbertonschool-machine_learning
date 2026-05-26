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
    # 1. Weighted_Price sütununu silirik
    df = df.drop(columns=['Weighted_Price'])

    # 2. Close sütununu əvvəlki sətrin dəyəri ilə doldururuq (forward fill)
    df['Close'] = df['Close'].ffill()

    # 3. High, Low və Open sütunlarındakı NaN-ları eyni sətirdəki Close ilə doldururuq
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Open'] = df['Open'].fillna(df['Close'])

    # 4. Həcm sütunlarındakı boşluqları 0 edirik
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

    return df
