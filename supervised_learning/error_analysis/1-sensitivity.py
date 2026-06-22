#!/usr/bin/env python3
"""
Defines a function that calculates the sensitivity for each class
"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix

    Parameters:
        confusion (numpy.ndarray): Confusion matrix of shape (classes, classes)
                                   where rows are correct labels and
                                   columns are predicted labels

    Returns:
        numpy.ndarray: Sensitivity values for each class of shape (classes,)
    """
    tp = np.diag(confusion)
    actual_positives = np.sum(confusion, axis=1)
    return tp / actual_positives
