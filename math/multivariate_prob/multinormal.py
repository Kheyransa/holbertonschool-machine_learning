#!/usr/bin/env python3
"""Contains the MultiNormal class for multivariate probability"""
import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution"""

    def __init__(self, data):
        """Initializes the MultiNormal instance with data"""
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        data_centered = data - self.mean
        self.cov = np.dot(data_centered, data_centered.T) / (n - 1)

    def pdf(self, x):
        """Calculates the PDF at a given data point"""
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]

        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)

        x_centered = x - self.mean

        norm_factor = 1.0 / np.sqrt(((2 * np.pi) ** d) * det)
        exponent = -0.5 * np.dot(np.dot(x_centered.T, inv), x_centered)
        pdf_val = norm_factor * np.exp(exponent)

        return float(pdf_val[0][0])
