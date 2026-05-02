#!/usr/bin/env python3
"""Adds two 2D matrices element-wise"""


def add_matrices2D(mat1, mat2):
    """Adds two 2D matrices element-wise and returns a new matrix"""
    # Check if rows are same
    if len(mat1) != len(mat2):
        return None
    # Check if columns are same
    if len(mat1[0]) != len(mat2[0]):
        return None

    new_matrix = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat1[0])):
            row.append(mat1[i][j] + mat2[i][j])
        new_matrix.append(row)

    return new_matrix
