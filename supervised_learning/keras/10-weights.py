#!/usr/bin/env python3
""" Modul 10-weights """
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """ Modelin çəkilərini fayla saxlayır """
    network.save_weights(filename)

    return None


def load_weights(network, filename):
    """ Fayldan modelə çəkiləri yükləyir """
    network.load_weights(filename)

    return None
