#!/usr/bin/env python3
"""
Defines a function that calculates the F1 score for each class
"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score for each class in a confusion matrix

    Parameters:
        confusion (numpy.ndarray): Confusion matrix of shape (classes, classes)
                                   where rows are correct labels and
                                   columns are predicted labels

    Returns:
        numpy.ndarray: F1 score values for each class of shape (classes,)
    """
    p = precision(confusion)
    s = sensitivity(confusion)

    return 2 * (p * s) / (p + s)
