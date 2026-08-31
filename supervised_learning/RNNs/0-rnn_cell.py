#!/usr/bin/env python3
"""Defines the RNNCell class that represents a cell of a simple RNN."""
import numpy as np


class RNNCell:
    """Represents a cell of a simple RNN."""

    def __init__(self, i, h, o):
        """Class constructor.

        Args:
            i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        """
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Performs forward propagation for one time step.

        Args:
            h_prev: numpy.ndarray of shape (m, h) containing the
                previous hidden state
            x_t: numpy.ndarray of shape (m, i) that contains the data
                input for the cell

        Returns:
            h_next, y: the next hidden state and the output of the cell
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concat, self.Wh) + self.bh)

        y_raw = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y_raw) / np.sum(np.exp(y_raw), axis=1, keepdims=True)

        return h_next, y
