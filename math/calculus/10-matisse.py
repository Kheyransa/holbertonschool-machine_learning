#!/usr/bin/env python3
"""Polynomial derivative module"""


def poly_derivative(poly):
    """Calculates the derivative of a polynomial"""
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    for x in poly:
        if not isinstance(x, (int, float)):
            return None

    if len(poly) == 1:
        return [0]

    # Derivative logic: coefficient * power, power drops by 1
    # Index 0 (constant) disappears, so we start from index 1
    derivative = [poly[i] * i for i in range(1, len(poly))]

    return derivative
