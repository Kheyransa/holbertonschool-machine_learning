#!/usr/bin/env python3
"""
Module to calculate the inverse of a square matrix
"""


def determinant(matrix):
    """
    Calculates the determinant of a matrix recursively
    """
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for c in range(len(matrix)):
        sub_matrix = [row[:c] + row[c + 1:] for row in matrix[1:]]
        det += ((-1) ** c) * matrix[0][c] * determinant(sub_matrix)
    return det


def inverse(matrix):
    """
    Calculates the inverse of matrix
    """
    if not isinstance(matrix, list) or not all(isinstance(row, list)
                                               for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Step 1: Check if the matrix is singular
    det = determinant(matrix)
    if det == 0:
        return None

    # Step 2: Handle 1x1 matrix case
    if n == 1:
        return [[1 / matrix[0][0]]]

    # Step 3: Calculate Adjugate (Transpose of Cofactors)
    adjugate_matrix = []
    for j in range(n):
        adj_row = []
        for i in range(n):
            # Minor submatrix
            sub_matrix = [
                row[:j] + row[j + 1:]
                for k, row in enumerate(matrix) if k != i
            ]
            # Cofactor and Transpose in one step
            cofactor = ((-1) ** (i + j)) * determinant(sub_matrix)
            # Divide by determinant to get Inverse element
            adj_row.append(cofactor / det)
        adjugate_matrix.append(adj_row)

    return adjugate_matrix
