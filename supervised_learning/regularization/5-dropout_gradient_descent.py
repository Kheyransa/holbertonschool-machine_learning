#!/usr/bin/env python3
"""Gradient Descent with Dropout"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization
    using gradient descent

    Y is a one-hot numpy.ndarray of shape (classes, m) with correct labels
    weights is a dictionary of the weights and biases of the neural network
    cache is a dictionary of the outputs and dropout masks of each layer
    alpha is the learning rate
    keep_prob is the probability that a node will be kept
    L is the number of layers of the network

    All layers use the tanh activation function except the last, which
    uses the softmax activation function
    The weights of the network are updated in place
    """
    m = Y.shape[1]
    weights_copy = weights.copy()
    dZ = cache['A' + str(L)] - Y

    for layer in range(L, 0, -1):
        A_prev = cache['A' + str(layer - 1)]
        W = weights_copy['W' + str(layer)]
        b = weights_copy['b' + str(layer)]

        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if layer > 1:
            dA_prev = np.matmul(W.T, dZ)
            dA_prev *= cache['D' + str(layer - 1)]
            dA_prev /= keep_prob
            dZ = dA_prev * (1 - A_prev ** 2)

        weights['W' + str(layer)] = W - alpha * dW
        weights['b' + str(layer)] = b - alpha * db
