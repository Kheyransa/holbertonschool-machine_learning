#!/usr/bin/env python3
"""Module that defines a Neuron class for binary classification"""

import numpy as np


class Neuron:
    """Defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """
        Initializes the Neuron

        Args:
            nx (int): number of input features to the neuron

        Raises:
            TypeError: if nx is not an integer
            ValueError: if nx is less than 1

        Attributes:
            W: weights vector, initialized with random normal distribution
            b: bias, initialized to 0
            A: activated output, initialized to 0
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.normal(size=(1, nx))
        self.b = 0
        self.A = 0
