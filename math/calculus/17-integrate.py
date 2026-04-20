#!/usr/bin/env python3
"""Polynomial integral module"""


def poly_integral(poly, C=0):
    """Calculates the integral of a polynomial"""
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not isinstance(C, int):
        return None

    integral = [C]
    for i in range(len(poly)):
        if not isinstance(poly[i], (int, float)):
            return None

        # Power rule: coefficient / (current_power + 1)
        value = poly[i] / (i + 1)

        # If the result is a whole number, represent it as an integer
        if value % 1 == 0:
            value = int(value)

        integral.append(value)

    # Remove trailing zeros to keep the list as small as possible
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
