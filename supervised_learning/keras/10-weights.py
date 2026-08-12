#!/usr/bin/env python3
"""Save and load weights."""

import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """Save a model's weights."""
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """Load a model's weights."""
    network.load_weights(filename)
