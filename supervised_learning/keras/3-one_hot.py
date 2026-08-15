#!/usr/bin/env python3
"""Converts a label vector into a one-hot matrix"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix

    labels: numpy.ndarray containing the labels to convert
    classes: number of classes to use for the one-hot matrix

    Returns: the one-hot matrix
    """
    return K.utils.to_categorical(labels, num_classes=classes)
