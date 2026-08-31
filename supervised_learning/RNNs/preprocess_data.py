#!/usr/bin/env python3
"""
preprocess_data.py

Preprocesses the raw Coinbase / Bitstamp BTC datasets so they can be
used to train an RNN that predicts the closing price of BTC one hour
into the future, given the previous 24 hours of data.

Usage:
    ./preprocess_data.py

Produces (in the current directory):
    train.npz, val.npz   -- each containing X (windows) and y (targets)
    scaler_params.npz    -- mean/std used for scaling (needed to invert
                             predictions later if desired)
"""
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
WINDOW_SIZE = 24 * 60   # 24 hours of 1-minute rows used as input
HORIZON = 60            # predict the close price 1 hour (60 rows) ahead
STEP = 60               # stride between windows so we get an "hourly"
                         # cadence of samples instead of a new sample
                         # every single minute

FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume_(BTC)",
                    "Volume_(Currency)", "Weighted_Price"]
TARGET_COLUMN = "Close"


def load_and_clean(path):
    """
    Loads one raw CSV and cleans it:
      - parses the Unix timestamp
      - drops rows with no trading activity
      - forward-fills any remaining sparse NaNs
    """
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")
    df = df.sort_values("Timestamp").reset_index(drop=True)

    df = df.dropna(subset=["Close"])

    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].ffill()
    df = df.dropna(subset=FEATURE_COLUMNS)

    return df


def restrict_to_recent(df, years_back=2):
    """
    Restricts data to the most recent, densely-traded period since
    the earliest years are too sparse to be representative.
    """
    cutoff = df["Timestamp"].max() - pd.Timedelta(days=365 * years_back)
    return df[df["Timestamp"] >= cutoff].reset_index(drop=True)


def scale(df, mean=None, std=None):
    """
    Standardizes (z-score) all feature columns. If mean/std are not
    provided they are computed from this dataframe (call on TRAIN
    split first, reuse returned mean/std on validation to avoid
    leakage).
    """
    if mean is None or std is None:
        mean = df[FEATURE_COLUMNS].mean()
        std = df[FEATURE_COLUMNS].std()
    scaled = df.copy()
    scaled[FEATURE_COLUMNS] = (df[FEATURE_COLUMNS] - mean) / std
    return scaled, mean, std


def make_windows(df):
    """
    Builds sliding windows of shape (WINDOW_SIZE, num_features) with
    the target being the Close price HORIZON steps after the end of
    the window.
    """
    features = df[FEATURE_COLUMNS].values
    target = df[TARGET_COLUMN].values

    X, y = [], []
    last_start = len(df) - WINDOW_SIZE - HORIZON
    for start in range(0, last_start, STEP):
        end = start + WINDOW_SIZE
        X.append(features[start:end])
        y.append(target[end + HORIZON - 1])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def main():
    coinbase = load_and_clean("coinbaseUSD_1-min_data.csv")
    bitstamp = load_and_clean("bitstampUSD_1-min_data.csv")

    df = restrict_to_recent(coinbase, years_back=2)

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)

    train_df, mean, std = scale(train_df)
    val_df, _, _ = scale(val_df, mean, std)

    X_train, y_train = make_windows(train_df)
    X_val, y_val = make_windows(val_df)

    close_mean, close_std = mean[TARGET_COLUMN], std[TARGET_COLUMN]
    y_train = (y_train - close_mean) / close_std
    y_val = (y_val - close_mean) / close_std

    np.savez("train.npz", X=X_train, y=y_train)
    np.savez("val.npz", X=X_val, y=y_val)
    np.savez("scaler_params.npz",
              mean=mean.values, std=std.values,
              close_mean=close_mean, close_std=close_std,
              columns=np.array(FEATURE_COLUMNS))

    print(f"Train windows: {X_train.shape}, Val windows: {X_val.shape}")


if __name__ == "__main__":
    main()
