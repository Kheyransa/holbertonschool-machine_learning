#!/usr/bin/env python3
"""Initializes cluster centroids for K-means."""

import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means.

    X is a numpy.ndarray of shape (n, d)
    k is a positive integer containing the number of clusters

    Returns:
        A numpy.ndarray of shape (k, d), or None on failure.
    """

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None

    if not isinstance(k, int) or k <= 0:
        return None

    # Find minimum and maximum values along each dimension
    min_values = np.min(X, axis=0)
    max_values = np.max(X, axis=0)

    # Generate k random centroids
    centroids = np.random.uniform(
        min_values,
        max_values,
        size=(k, X.shape[1])
    )

    return centroids
