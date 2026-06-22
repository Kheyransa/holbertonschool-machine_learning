#!/usr/bin/env python3
"""
Defines a function that creates a confusion matrix
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix from one-hot encoded labels and logits

    Parameters:
        labels (numpy.ndarray): One-hot encoded correct labels of shape
                                (m, classes)
        logits (numpy.ndarray): One-hot encoded predicted labels of shape
                                (m, classes)

    Returns:
        numpy.ndarray: Confusion matrix of shape (classes, classes)
    """
    return np.matmul(labels.T, logits)
