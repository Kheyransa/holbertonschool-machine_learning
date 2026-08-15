#!/usr/bin/env python3
""" Modul 9-model """
import tensorflow.keras as K


def save_model(network, filename):
    """ Bütün modeli fayla saxlayır """
    network.save(filename)

    return None


def load_model(filename):
    """ Fayldan bütün modeli yükləyir """
    network = K.models.load_model(filename)

    return network
