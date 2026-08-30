#!/usr/bin/env python3
"""Module that initializes variables for a Gaussian Mixture Model"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model

    X: numpy.ndarray of shape (n, d), the data set
    k: positive integer, number of clusters

    Returns: pi, m, S, or None, None, None on failure
        pi: numpy.ndarray of shape (k,), priors for each cluster
        m: numpy.ndarray of shape (k, d), centroid means (from K-means)
        S: numpy.ndarray of shape (k, d, d), covariance matrices (identity)
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None

    n, d = X.shape

    pi = np.full((k,), 1 / k)

    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None

    S = np.tile(np.identity(d), (k, 1, 1))

    return pi, m, S
