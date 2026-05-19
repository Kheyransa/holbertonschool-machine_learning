#!/usr/bin/env python3
"""Contains the continuous posterior function for Bayesian probability"""
import numpy as np
from scipy import special


def posterior(x, n, p1, p2):
    """Calculates the posterior probability for a continuous range"""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        msg = "x must be an integer that is greater than or equal to 0"
        raise ValueError(msg)

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(p1, float) or not (0 <= p1 <= 1):
        raise ValueError("p1 must be a float in the range [0, 1]")

    if not isinstance(p2, float) or not (0 <= p2 <= 1):
        raise ValueError("p2 must be a float in the range [0, 1]")

    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    # Beta paylanmasının parametrləri
    alpha = x + 1
    beta = n - x + 1

    # p2 və p1 nöqtələrində Beta CDF (betainc) hesablanması
    cdf_p2 = special.betainc(alpha, beta, p2)
    cdf_p1 = special.betainc(alpha, beta, p1)

    # Test faylının .round() metodunu dəstəkləməsi üçün np.float64 istifadə edirik
    return np.float64(cdf_p2 - cdf_p1)
