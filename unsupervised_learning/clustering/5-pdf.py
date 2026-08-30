#!/usr/bin/env python3
"""Module that calculates the PDF of a Gaussian distribution"""
import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution

    X: numpy.ndarray of shape (n, d), data points to evaluate
    m: numpy.ndarray of shape (d,), mean of the distribution
    S: numpy.ndarray of shape (d, d), covariance of the distribution

    Returns: P, numpy.ndarray of shape (n,) containing PDF values,
             or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None

    n, d = X.shape
    if m.shape[0] != d or S.shape[0] != d or S.shape[1] != d:
        return None

    det = np.linalg.det(S)
    inv = np.linalg.inv(S)

    X_m = X - m

    denom = np.sqrt(((2 * np.pi) ** d) * det)

    exponent = -0.5 * np.sum(X_m @ inv * X_m, axis=1)

    P = np.exp(exponent) / denom

    P = np.maximum(P, 1e-300)

    return P
