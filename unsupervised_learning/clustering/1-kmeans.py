#!/usr/bin/env python3
"""Module that performs K-means on a dataset"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset

    X: numpy.ndarray of shape (n, d) containing the dataset
    k: positive integer, number of clusters
    iterations: positive integer, max number of iterations

    Returns: C, clss, or None, None on failure
        C: numpy.ndarray of shape (k, d), centroid means
        clss: numpy.ndarray of shape (n,), cluster index for each point
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    low = X.min(axis=0)
    high = X.max(axis=0)
    C = np.random.uniform(low, high, size=(k, d))

    for i in range(iterations):
        C_prev = np.copy(C)

        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=-1)
        clss = np.argmin(distances, axis=1)

        for j in range(k):
            if np.sum(clss == j) == 0:
                C[j] = np.random.uniform(low, high, size=(1, d))
            else:
                C[j] = X[clss == j].mean(axis=0)

        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=-1)
        clss = np.argmin(distances, axis=1)

        if np.array_equal(C, C_prev):
            return C, clss

    return C, clss
