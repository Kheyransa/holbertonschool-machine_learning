#!/usr/bin/env python3
"""Performs element-wise operations using NumPy"""
import numpy as np


def np_elementwise(mat1, mat2):
    """Returns a tuple containing element-wise sum, diff, prod, and quot"""
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
