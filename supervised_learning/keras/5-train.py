#!/usr/bin/env python3
"""
Train a model using mini-batch gradient descent.
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    Trains a neural network.

    Args:
        network: the model to train
        data: input data
        labels: one-hot labels
        batch_size: mini-batch size
        epochs: number of epochs
        validation_data: data to validate with
        verbose: whether to print progress
        shuffle: whether to shuffle data

    Returns:
        The History object generated after training.
    """
    return network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        verbose=verbose,
        shuffle=shuffle
    )
