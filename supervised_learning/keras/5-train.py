#!/usr/bin/env python3
"""
Train a model using mini-batch gradient descent
"""

def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    Trains a neural network.

    Args:
        network: the model to train
        data: numpy.ndarray of shape (m, nx) containing the input data
        labels: one-hot numpy.ndarray of shape (m, classes)
        batch_size: size of the mini-batches
        epochs: number of epochs to train
        validation_data: data to validate the model with
        verbose: determines if output should be printed during training
        shuffle: determines whether to shuffle the data every epoch

    Returns:
        The History object generated after training.
    """
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        verbose=verbose,
        shuffle=shuffle
    )

    return history
