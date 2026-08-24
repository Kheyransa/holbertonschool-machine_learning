#!/usr/bin/env python3
"""Performs a t-SNE transformation."""

import numpy as np

pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0,
         iterations=1000, lr=500):
    """
    Performs a t-SNE transformation.

    X is a numpy.ndarray of shape (n, d) containing the dataset
    to be transformed by t-SNE.

    ndims is the new dimensional representation of X.
    idims is the intermediate dimensional representation of X
    after PCA.
    perplexity is the perplexity.
    iterations is the number of iterations.
    lr is the learning rate.

    Returns:
        Y, a numpy.ndarray of shape (n, ndims) containing the
        optimized low dimensional transformation of X.
    """

    n = X.shape[0]

    # Reduce X using PCA
    X = pca(X, idims)

    # Calculate P affinities
    P = P_affinities(X, perplexity=perplexity)

    # Early exaggeration
    P = P * 4

    # Initialize Y
    Y = np.random.normal(0, 1, (n, ndims))

    # Previous Y
    Y_prev = Y.copy()

    for i in range(iterations):

        # Calculate gradients and Q affinities
        dY, Q = grads(Y, P)

        # Momentum
        if i < 20:
            a = 0.5
        else:
            a = 0.8

        # Gradient descent
        Y_new = Y + lr * dY + a * (Y - Y_prev)

        # Update previous Y
        Y_prev = Y

        # Update Y
        Y = Y_new

        # Re-center Y
        Y = Y - np.mean(Y, axis=0)

        # Print cost every 100 iterations
        if (i + 1) % 100 == 0:
            C = cost(P, Q)
            print("Cost at iteration {}: {}".format(i + 1, C))

        # End early exaggeration
        if i == 99:
            P = P / 4

    return Y
