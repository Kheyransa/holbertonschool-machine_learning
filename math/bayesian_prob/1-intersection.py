#!/usr/bin/env python3
"""Contains the intersection function for Bayesian probability"""
import numpy as np


def intersection(x, n, P, Pr):
    """Calculates the intersection of obtaining data with various priors"""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        msg = "x must be an integer that is greater than or equal to 0"
        raise ValueError(msg)

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Faktorial hesablamaq üçün daxili köməkçi funksiya
    def fact(num):
        f = 1
        for i in range(1, num + 1):
            f *= i
        return f

    # Likelihood üçün kombinasiya hissəsi: n! / (x! * (n - x)!)
    n_fact = fact(n)
    x_fact = fact(x)
    nx_fact = fact(n - x)
    combination = n_fact / (x_fact * nx_fact)

    # Likelihood massivinin hesablanması
    likelihood = combination * (P ** x) * ((1 - P) ** (n - x))

    # Intersection = Likelihood * Prior
    intersection_value = likelihood * Pr

    return intersection_value
