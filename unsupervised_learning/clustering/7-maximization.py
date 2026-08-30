#!/usr/bin/env python3
"""Module that calculates the maximization step in the EM algorithm"""
import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM

    X: numpy.ndarray of shape (n, d), the data set
    g: numpy.ndarray of shape (k, n), posterior probabilities

    Returns: pi, m, S, or None, None, None on failure
        pi: numpy.ndarray of shape (k,), updated priors
        m: numpy.ndarray of shape (k, d), updated centroid means
        S: numpy.ndarray of shape (k, d, d), updated covariance matrices
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    n, d = X.shape
    k = g.shape[0]

    if g.shape[1] != n:
        return None, None, None

    sums = np.sum(g, axis=0)
    if not np.all(np.isclose(sums, 1)):
        return None, None, None

    Nk = np.sum(g, axis=1)

    pi = Nk / n

    m = np.matmul(g, X) / Nk[:, np.newaxis]

    S = np.zeros((k, d, d))

    for i in range(k):
        X_m = X - m[i]
        S[i] = np.matmul(g[i] * X_m.T, X_m) / Nk[i]

    return pi, m, S
