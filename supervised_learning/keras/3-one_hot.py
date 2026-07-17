#!/usr/bin/env python3
"""
Converts labels to a one-hot matrix.
"""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.

    Args:
        labels: label vector
        classes: number of classes

    Returns:
        One-hot matrix.
    """
    return K.utils.to_categorical(labels, num_classes=classes)
