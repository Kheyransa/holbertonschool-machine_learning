#!/usr/bin/env python3
"""
Module to calculate the adjugate matrix of a square matrix
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


def adjugate(matrix):
    """
    Calculates the adjugate matrix of matrix
    """
    if not isinstance(matrix, list) or not all(isinstance(row, list)
                                               for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return [[1]]

    # Step 1: Calculate the Cofactor Matrix
    cofactor_matrix = []
    for i in range(n):
        row_cofactors = []
        for j in range(n):
            sub_matrix = [
                row[:j] + row[j + 1:]
                for k, row in enumerate(matrix) if k != i
            ]
            minor_val = determinant(sub_matrix)
            row_cofactors.append(((-1) ** (i + j)) * minor_val)
        cofactor_matrix.append(row_cofactors)

    # Step 2: Transpose the Cofactor Matrix to get the Adjugate
    adjugate_matrix = []
    for j in range(n):
        new_row = []
        for i in range(n):
            new_row.append(cofactor_matrix[i][j])
        adjugate_matrix.append(new_row)

    return adjugate_matrix
