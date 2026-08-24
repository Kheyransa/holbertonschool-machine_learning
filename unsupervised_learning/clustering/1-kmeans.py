#!/usr/bin/env python3
"""Performs K-means on a dataset."""

import numpy as np

initialize = __import__('0-initialize').initialize


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset.

    X is a numpy.ndarray of shape (n, d)
    k is a positive integer containing the number of clusters
    iterations is the maximum number of iterations

    Returns:
        C, clss
    """

    # Validate X
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    # Validate k
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None

    # Validate iterations
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    # Initialize centroids
    C = initialize(X, k)

    # Minimum and maximum values of each dimension
    min_values = np.min(X, axis=0)
    max_values = np.max(X, axis=0)

    for _ in range(iterations):

        # Calculate distance from each point to each centroid
        distances = np.linalg.norm(
            X[:, np.newaxis, :] - C[np.newaxis, :, :],
            axis=2
        )

        # Assign each point to its closest centroid
        clss = np.argmin(distances, axis=1)

        # Save old centroids
        C_old = C.copy()

        # Update centroids
        for j in range(k):
            points = X[clss == j]

            if points.shape[0] == 0:
                C[j] = np.random.uniform(
                    min_values,
                    max_values
                )
            else:
                C[j] = np.mean(points, axis=0)

        # Stop if centroids did not change
        if np.array_equal(C, C_old):
            return C, clss

    return C, clss
