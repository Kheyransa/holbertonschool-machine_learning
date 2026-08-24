#!/usr/bin/env python3
"""Performs PCA on a dataset."""

import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset.

    X is a numpy.ndarray of shape (n, d)
    ndim is the new dimensionality of the transformed X

    Returns:
        T: numpy.ndarray of shape (n, ndim)
    """

    # Perform Singular Value Decomposition
    u, s, vh = np.linalg.svd(X, full_matrices=False)

    # Select the first ndim principal components
    W = vh[:ndim].T

    # Transform X into the new space
    T = np.matmul(X, W)

    return T
