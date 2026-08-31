#!/usr/bin/env python3
"""Defines a function that performs forward propagation for a simple RNN."""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """Performs forward propagation for a simple RNN.

    Args:
        rnn_cell: an instance of RNNCell used for the forward propagation
        X: numpy.ndarray of shape (t, m, i) with the data to be used
            t is the maximum number of time steps
            m is the batch size
            i is the dimensionality of the data
        h_0: numpy.ndarray of shape (m, h) with the initial hidden state
            h is the dimensionality of the hidden state

    Returns:
        H, Y: H is a numpy.ndarray containing all of the hidden states,
            Y is a numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    h = h_0.shape[1]

    H = np.zeros((t + 1, m, h))
    H[0] = h_0

    Y = []
    h_prev = h_0
    for step in range(t):
        h_next, y = rnn_cell.forward(h_prev, X[step])
        H[step + 1] = h_next
        Y.append(y)
        h_prev = h_next

    Y = np.array(Y)

    return H, Y
