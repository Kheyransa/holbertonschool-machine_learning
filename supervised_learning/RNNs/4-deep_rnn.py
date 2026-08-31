#!/usr/bin/env python3
"""Defines a function that performs forward propagation for a deep RNN."""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Performs forward propagation for a deep RNN.

    Args:
        rnn_cells: list of RNNCell instances of length l that will be
            used for the forward propagation
            l is the number of layers
        X: numpy.ndarray of shape (t, m, i) with the data to be used
            t is the maximum number of time steps
            m is the batch size
            i is the dimensionality of the data
        h_0: numpy.ndarray of shape (l, m, h) with the initial hidden
            state
            h is the dimensionality of the hidden state

    Returns:
        H, Y: H is a numpy.ndarray containing all of the hidden states,
            Y is a numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    length, _, h = h_0.shape

    H = np.zeros((t + 1, length, m, h))
    H[0] = h_0

    Y = []
    for step in range(t):
        x = X[step]
        for layer in range(length):
            h_prev = H[step, layer]
            h_next, y = rnn_cells[layer].forward(h_prev, x)
            H[step + 1, layer] = h_next
            x = h_next
        Y.append(y)

    Y = np.array(Y)

    return H, Y
