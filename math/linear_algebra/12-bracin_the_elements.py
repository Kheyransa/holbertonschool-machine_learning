#!/usr/bin/env python3
"""Performs element-wise operations without explicit import"""


def np_elementwise(mat1, mat2):
    """Returns a tuple containing element-wise sum, diff, prod, and quot"""
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
