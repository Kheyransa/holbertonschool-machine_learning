#!/usr/bin/env python3

import numpy as np

pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0,
         iterations=1000, lr=500):
    """
    Performs a t-SNE transformation.
    """

    # 1. Reduce X using PCA
    X = pca(X, idims)

    # 2. Calculate high-dimensional affinities
    P = P_affinities(X, perplexity)

    # 3. Initialize low-dimensional representation
    n = X.shape[0]
    Y = np.random.normal(0, 1, (n, ndims))

    # 4. Previous Y for momentum
    Y_prev = np.zeros_like(Y)

    # 5. Gradient descent
    for iteration in range(1, iterations + 1):

        # Early exaggeration
        if iteration <= 100:
            P_current = 4 * P
        else:
            P_current = P

        # Momentum
        if iteration <= 20:
            alpha = 0.5
        else:
            alpha = 0.8

        # Calculate gradient
        # Check your 6-grads.py signature here
        dY, Q = grads(Y, P_current)

        # Gradient descent update
        Y_new = Y - lr * dY + alpha * (Y - Y_prev)

        # Save current Y
        Y_prev = Y

        # Update Y
        Y = Y_new

        # Re-center
        Y = Y - np.mean(Y, axis=0)

        # Cost every 100 iterations
        if iteration % 100 == 0:
            C = cost(P_current, Q)
            print("Cost at iteration {}: {}".format(iteration, C))

    return Y
