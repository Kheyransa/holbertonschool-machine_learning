#!/usr/bin/env python3
"""Defines the LSTMCell class that represents an LSTM unit."""
import numpy as np


def sigmoid(x):
    """Computes the sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


class LSTMCell:
    """Represents an LSTM unit."""

    def __init__(self, i, h, o):
        """Class constructor.

        Args:
            i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        """
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """Performs forward propagation for one time step.

        Args:
            h_prev: numpy.ndarray of shape (m, h) containing the
                previous hidden state
            c_prev: numpy.ndarray of shape (m, h) containing the
                previous cell state
            x_t: numpy.ndarray of shape (m, i) that contains the data
                input for the cell

        Returns:
            h_next, c_next, y: the next hidden state, the next cell
                state, and the output of the cell
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        f_t = sigmoid(np.matmul(concat, self.Wf) + self.bf)
        u_t = sigmoid(np.matmul(concat, self.Wu) + self.bu)
        c_tilde = np.tanh(np.matmul(concat, self.Wc) + self.bc)
        c_next = f_t * c_prev + u_t * c_tilde

        o_t = sigmoid(np.matmul(concat, self.Wo) + self.bo)
        h_next = o_t * np.tanh(c_next)

        y_raw = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y_raw) / np.sum(np.exp(y_raw), axis=1, keepdims=True)

        return h_next, c_next, y
