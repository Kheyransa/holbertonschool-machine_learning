#!/usr/bin/env python3
"""Concatenates two matrices along a specific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenates two 2D matrices and returns a new matrix"""
    if axis == 0:
        # Check if columns are the same
        if len(mat1[0]) != len(mat2[0]):
            return None
        # Create a new matrix with copies of rows from mat1 and mat2
        return [row[:] for row in mat1] + [row[:] for row in mat2]

    if axis == 1:
        # Check if rows are the same
        if len(mat1) != len(mat2):
            return None
        # Concatenate each row of mat1 with corresponding row of mat2
        return [mat1[i] + mat2[i] for i in range(len(mat1))]

    return None
