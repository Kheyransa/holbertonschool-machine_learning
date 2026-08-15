#!/usr/bin/env python3
""" Modul 11-config """
import tensorflow.keras as K


def save_config(network, filename):
    """ Modelin konfiqurasiyasını JSON formatında saxlayır """
    config = network.to_json()
    with open(filename, 'w') as f:
        f.write(config)

    return None


def load_config(filename):
    """ JSON konfiqurasiya faylından modeli yükləyir """
    with open(filename, 'r') as f:
        config = f.read()
    network = K.models.model_from_json(config)

    return network
