#!/usr/bin/env python3
"""Transpose a 2D matrix"""


def matrix_transpose(matrix):
    """Returns the transpose of a 2D matrix"""
    # Number of columns in original becomes number of rows in transpose
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
