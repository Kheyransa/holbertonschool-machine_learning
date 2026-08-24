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

    # Validate input
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    # Initialize centroids
    C = initialize(X, k)

    # Values used to reinitialize empty clusters
    min_values = np.min(X, axis=0)
    max_values = np.max(X, axis=0)

    # Generate replacement centroids
    empty_centroids = np.random.uniform(
        min_values,
        max_values,
        size=(k, X.shape[1])
    )

    for _ in range(iterations):

        # Calculate distances from every point to every centroid
        distances = np.linalg.norm(
            X[:, np.newaxis, :] - C[np.newaxis, :, :],
            axis=2
        )

        # Assign every point to its closest centroid
        clss = np.argmin(distances, axis=1)

        # Store old centroids
        C_old = C.copy()

        # Update each centroid
        for j in range(k):
            points = X[clss == j]

            if points.shape[0] == 0:
                C[j] = empty_centroids[j]
            else:
                C[j] = np.mean(points, axis=0)

        # Stop if centroids didn't change
        if np.array_equal(C, C_old):
            break

    return C, clss
