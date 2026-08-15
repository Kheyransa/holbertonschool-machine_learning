#!/usr/bin/env python3
""" Modul 13-predict """
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """ Neyron şəbəkəsi ilə proqnoz verir """
    prediction = network.predict(data, verbose=verbose)

    return prediction
