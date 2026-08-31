#!/usr/bin/env python3
"""Convolutional Autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder

    input_dims: tuple of integers, dimensions of the model input
    filters: list, number of filters for each convolutional layer
             in the encoder (reversed for the decoder)
    latent_dims: tuple of integers, dimensions of the latent
                 space representation

    Returns: encoder, decoder, auto
    """
    # ------------------ Encoder ------------------
    encoder_inputs = keras.Input(shape=input_dims)
    x = encoder_inputs
    for f in filters:
        x = keras.layers.Conv2D(
            f, (3, 3), padding='same', activation='relu')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    encoder = keras.Model(inputs=encoder_inputs, outputs=x)

    # ------------------ Decoder ------------------
    decoder_inputs = keras.Input(shape=latent_dims)
    x = decoder_inputs
    for i in range(len(filters) - 1, 0, -1):
        x = keras.layers.Conv2D(
            filters[i], (3, 3), padding='same', activation='relu')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)

    x = keras.layers.Conv2D(
        filters[0], (3, 3), padding='valid', activation='relu')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)

    decoder_outputs = keras.layers.Conv2D(
        input_dims[-1], (3, 3), padding='same', activation='sigmoid')(x)

    decoder = keras.Model(inputs=decoder_inputs, outputs=decoder_outputs)

    # ------------------ Autoencoder ------------------
    auto_outputs = decoder(encoder(encoder_inputs))
    auto = keras.Model(inputs=encoder_inputs, outputs=auto_outputs)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
