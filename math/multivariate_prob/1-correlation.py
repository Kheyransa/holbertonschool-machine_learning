#!/usr/bin/env python3
"""Contains the correlation function for multivariate probability"""
import numpy as np


def correlation(C):
    """Calculates a correlation matrix based on a covariance matrix"""
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    diag = np.diag(C)
    std_dev = np.sqrt(diag)
    outer_std = np.outer(std_dev, std_dev)
    corr = C / outer_std

    return corr
