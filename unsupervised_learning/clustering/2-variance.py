#!/usr/bin/env python3
"""Module that calculates the total intra-cluster variance for a data set"""
import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a data set

    X: numpy.ndarray of shape (n, d), the data set
    C: numpy.ndarray of shape (k, d), centroid means for each cluster

    Returns: var, the total variance, or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=-1)

    min_distances = np.min(distances, axis=1)

    var = np.sum(min_distances ** 2)

    return var
