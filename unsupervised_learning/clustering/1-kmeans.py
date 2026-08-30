#!/usr/bin/env python3
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d)
        k: positive integer, number of clusters
        iterations: positive integer, maximum number of iterations

    Returns:
        C: numpy.ndarray of shape (k, d), centroids
        clss: numpy.ndarray of shape (n,), cluster assignments
        or None, None on failure
    """

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    # Initialize centroids using a multivariate uniform distribution
    low = np.min(X, axis=0)
    high = np.max(X, axis=0)

    C = np.random.uniform(low, high, size=(k, X.shape[1]))

    for _ in range(iterations):
        # Calculate distances from every point to every centroid
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)

        # Assign each point to its closest centroid
        clss = np.argmin(distances, axis=1)

        # Calculate new centroids
        C_new = np.array([
            np.mean(X[clss == i], axis=0)
            if np.any(clss == i)
            else np.random.uniform(low, high, size=X.shape[1])
            for i in range(k)
        ])

        # Stop if centroids haven't changed
        if np.array_equal(C, C_new):
            return C, clss

        C = C_new

    return C, clss
