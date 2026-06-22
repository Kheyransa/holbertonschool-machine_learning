#!/usr/bin/env python3
"""
Defines a function that calculates the specificity for each class
"""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix

    Parameters:
        confusion (numpy.ndarray): Confusion matrix of shape (classes, classes)
                                   where rows are correct labels and
                                   columns are predicted labels

    Returns:
        numpy.ndarray: Specificity values for each class of shape (classes,)
    """
    total = np.sum(confusion)
    tp = np.diag(confusion)
    row_sums = np.sum(confusion, axis=1)
    col_sums = np.sum(confusion, axis=0)

    fp = col_sums - tp
    fn = row_sums - tp
    tn = total - (tp + fp + fn)

    actual_negatives = tn + fp
    return tn / actual_negatives
