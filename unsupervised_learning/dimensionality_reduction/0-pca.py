#!/usr/bin/env python3
"""Performs PCA on a dataset"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset.

    X is a numpy.ndarray of shape (n, d) where:
        n is the number of data points
        d is the number of dimensions in each point
        all dimensions have a mean of 0 across all data points

    var is the fraction of the variance that the PCA transformation
    should maintain.

    Returns:
        W: numpy.ndarray of shape (d, nd)
    """
    u, s, vh = np.linalg.svd(X)

    # Variance explained by each principal component
    explained_variance = s ** 2

    # Cumulative fraction of explained variance
    cum_var = np.cumsum(explained_variance) / np.sum(explained_variance)

    # Number of components needed
    nd = np.argwhere(cum_var >= var)[0, 0] + 1

    # Principal component directions
    W = vh[:nd].T

    return W
