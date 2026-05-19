#!/usr/bin/env python3
"""Contains the continuous posterior function for Bayesian probability"""
from scipy import special


class RoundableFloat(float):
    """A custom float subclass that supports the .round() method"""

    def round(self, ndigits=None):
        """Custom round method mimicking numpy's float round behavior"""
        return round(self, ndigits)


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

    alpha = x + 1
    beta = n - x + 1

    cdf_p2 = special.betainc(alpha, beta, p2)
    cdf_p1 = special.betainc(alpha, beta, p1)

    # Nəticəni xüsusi RoundableFloat tipində qaytarırıq
    return RoundableFloat(cdf_p2 - cdf_p1)
