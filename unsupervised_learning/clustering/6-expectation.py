#!/usr/bin/env python3
"""Module that calculates the expectation step in the EM algorithm for a GMM"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm for a GMM

    X: numpy.ndarray of shape (n, d), the data set
    pi: numpy.ndarray of shape (k,), priors for each cluster
    m: numpy.ndarray of shape (k, d), centroid means for each cluster
    S: numpy.ndarray of shape (k, d, d), covariance matrices for each cluster

    Returns: g, l, or None, None on failure
        g: numpy.ndarray of shape (k, n), posterior probabilities
        l: the total log likelihood
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if m.shape[0] != k or m.shape[1] != d:
        return None, None
    if S.shape[0] != k or S.shape[1] != d or S.shape[2] != d:
        return None, None
    if not np.isclose(np.sum(pi), 1):
        return None, None

    numerator = np.zeros((k, n))

    for i in range(k):
        P = pdf(X, m[i], S[i])
        if P is None:
            return None, None
        numerator[i] = pi[i] * P

    denominator = np.sum(numerator, axis=0)

    g = numerator / denominator

    log_likelihood = np.sum(np.log(denominator))

    return g, log_likelihood
