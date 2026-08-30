#!/usr/bin/env python3
"""Module that tests for the optimum number of clusters by variance"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance

    X: numpy.ndarray of shape (n, d), the data set
    kmin: positive integer, minimum number of clusters to check (inclusive)
    kmax: positive integer, maximum number of clusters to check (inclusive)
    iterations: positive integer, max number of iterations for K-means

    Returns: results, d_vars, or None, None on failure
        results: list of outputs of K-means for each cluster size
        d_vars: list of variance differences from the smallest cluster size
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None

    n, d = X.shape

    if kmax is None:
        kmax = n

    if not isinstance(kmax, int) or kmax <= 0:
        return None, None
    if kmin >= kmax:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    results = []
    variances = []

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None or clss is None:
            return None, None
        results.append((C, clss))
        variances.append(variance(X, C))

    d_vars = [variances[0] - var in variances]
    d_vars = [variances[0] - var for var in variances]

    return results, d_vars
