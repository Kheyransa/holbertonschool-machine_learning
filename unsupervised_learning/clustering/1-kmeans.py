#!/usr/bin/env python3
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d)
        k: positive integer containing the number of clusters
        iterations: positive integer containing maximum iterations

    Returns:
        C: numpy.ndarray of shape (k, d), containing centroid means
        clss: numpy.ndarray of shape (n,), containing cluster indexes
        None, None on failure
    """

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    min_values = np.min(X, axis=0)
    max_values = np.max(X, axis=0)

    C = np.random.uniform(
        min_values,
        max_values,
        (k, X.shape[1])
    )

    for _ in range(iterations):
        distances = np.linalg.norm(
            X[:, np.newaxis] - C,
            axis=2
        )

        clss = np.argmin(distances, axis=1)

        C_new = np.array([
            np.mean(X[clss == i], axis=0)
            if np.any(clss == i)
            else np.random.uniform(
                min_values,
                max_values,
                X.shape[1]
            )
            for i in range(k)
        ])

        if np.array_equal(C, C_new):
            break

        C = C_new

    return C, clss
