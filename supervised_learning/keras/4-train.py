#!/usr/bin/env python3
"""
Trains a Keras model.
"""

def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent.

    Args:
        network: Keras model
        data: input data
        labels: one-hot labels
        batch_size: batch size
        epochs: number of epochs
        verbose: verbosity mode
        shuffle: whether to shuffle the data

    Returns:
        The History object generated after training.
    """
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle
    )

    return history
