#!/usr/bin/env python3
"""Extract Word2Vec"""
import tensorflow as tf


def gensim_to_keras(model):
    """
    Converts a gensim word2vec model to a keras Embedding layer.

    model is a trained gensim word2vec model

    Returns: the trainable keras Embedding
    """
    keyed_vectors = model.wv
    weights = keyed_vectors.vectors

    layer = tf.keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True)

    return layer
