#!/usr/bin/env python3
"""Performs matrix multiplication"""


def mat_mul(mat1, mat2):
    """Multiplies two 2D matrices and returns a new matrix"""
    # Check if multiplication is possible (cols of mat1 == rows of mat2)
    if len(mat1[0]) != len(mat2):
        return None

    # Result matrix dimensions: rows of mat1 x cols of mat2
    res = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            element = 0
            for k in range(len(mat2)):
                element += mat1[i][k] * mat2[k][j]
            row.append(element)
        res.append(row)

    return res
