#!/usr/bin/env python3
"""Contains the likelihood function for Bayesian probability"""
import numpy as np


def likelihood(x, n, P):
    """Calculates the likelihood of obtaining data given various hypotheses"""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        msg = "x must be an integer that is greater than or equal to 0"
        raise ValueError(msg)

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Kombinasiyanın hesablanması: n! / (x! * (n - x)!)
    import math
    n_fact = math.factorial(n)
    x_fact = math.factorial(x)
    nx_fact = math.factorial(n - x)
    combination = n_fact / (x_fact * nx_fact)

    # Likelihood massivinin hesablanması: combination * P^x * (1 - P)^(n - x)
    l_value = combination * (P ** x) * ((1 - P) ** (n - x))

    return l_value
