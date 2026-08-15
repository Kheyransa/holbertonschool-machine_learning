#!/usr/bin/env python3
""" Modul 12-test """
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """ Neyron şəbəkəsini test edir """
    evaluation = network.evaluate(data, labels, verbose=verbose)

    return evaluation
