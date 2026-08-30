#!/usr/bin/env python3
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d)
            Dataset containing n data points with d dimensions.
        k: positive integer
            Number of clusters.
        iterations: positive integer
            Maximum number of iterations.

    Returns:
        C: numpy.ndarray of shape (k, d)
            Centroids of the clusters.
        clss: numpy.ndarray of shape (n,)
            Index of the cluster each data point belongs to.

        Returns None, None if the input is invalid.
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

    # Get minimum and maximum values for each dimension
    min_values = np.min(X, axis=0)
    max_values = np.max(X, axis=0)

    # Initialize centroids using multivariate uniform distribution
    C = np.random.uniform(
        min_values,
        max_values,
        size=(k, X.shape[1])
    )

    for _ in range(iterations):

        # Calculate distance between every point and every centroid
        distances = np.linalg.norm(
            X[:, np.newaxis] - C,
            axis=2
        )

        # Assign every point to its closest centroid
        clss = np.argmin(distances, axis=1)

        # Save old centroids for convergence check
        C_old = C.copy()

        # Calculate new centroids
        C = np.array([
            np.mean(X[clss == i], axis=0)
            if np.any(clss == i)
            else np.random.uniform(
                min_values,
                max_values,
                size=X.shape[1]
            )
            for i in range(k)
        ])

        # Stop if centroids have not changed
        if np.array_equal(C, C_old):
            break

    # Recalculate classes using the FINAL centroids
    distances = np.linalg.norm(
        X[:, np.newaxis] - C,
        axis=2
    )
    clss = np.argmin(distances, axis=1)

    return C, clss
