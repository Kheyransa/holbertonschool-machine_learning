#!/usr/bin/env python3
"""
forecast_btc.py

Builds, trains, and validates an RNN (LSTM) model that forecasts the
BTC closing price one hour ahead, using the preceding 24 hours of
1-minute data. Expects train.npz / val.npz produced by
preprocess_data.py to be present in the current directory.
"""
import numpy as np
import tensorflow as tf


BATCH_SIZE = 64
SHUFFLE_BUFFER = 1000
EPOCHS = 20


def make_dataset(path, shuffle):
    """Wraps a preprocessed .npz file into a tf.data.Dataset."""
    data = np.load(path)
    X, y = data["X"], data["y"]

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(SHUFFLE_BUFFER)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds, X.shape[1:]


def build_model(input_shape):
    """
    A stacked LSTM regressor:
      - LSTM layers capture temporal dependencies across the 24h window
      - Dropout for regularization
      - A single linear output unit for regression
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="mse",
                  metrics=["mae"])
    return model


def main():
    train_ds, input_shape = make_dataset("train.npz", shuffle=True)
    val_ds, _ = make_dataset("val.npz", shuffle=False)

    model = build_model(input_shape)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True),
    ]

    model.fit(train_ds,
              validation_data=val_ds,
              epochs=EPOCHS,
              callbacks=callbacks)

    val_loss, val_mae = model.evaluate(val_ds)
    print(f"Validation MSE: {val_loss:.6f}, Validation MAE: {val_mae:.6f}")

    model.save("btc_forecast_model.keras")


if __name__ == "__main__":
    main()
