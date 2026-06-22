#!/usr/bin/env python3
"""
Defines a function that calculates the precision for each class
"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix

    Parameters:
        confusion (numpy.ndarray): Confusion matrix of shape (classes, classes)
                                   where rows are correct labels and
                                   columns are predicted labels

    Returns:
        numpy.ndarray: Precision values for each class of shape (classes,)
    """
    tp = np.diag(confusion)
    predicted_positives = np.sum(confusion, axis=0)
    return tp / predicted_positives
