#!/usr/bin/env python3
"""Module that calculates a GMM from a dataset using sklearn"""
import sklearn.mixture


def gmm(X, k):
    """
    Calculates a GMM from a dataset using sklearn

    X: numpy.ndarray of shape (n, d), the dataset
    k: the number of clusters

    Returns: pi, m, S, clss, bic
        pi: numpy.ndarray of shape (k,), cluster priors
        m: numpy.ndarray of shape (k, d), centroid means
        S: numpy.ndarray of shape (k, d, d), covariance matrices
        clss: numpy.ndarray of shape (n,), cluster indices for each point
        bic: the BIC value of the model
    """
    model = sklearn.mixture.GaussianMixture(n_components=k).fit(X)

    pi = model.weights_
    m = model.means_
    S = model.covariances_
    clss = model.predict(X)
    bic = model.bic(X)

    return pi, m, S, clss, bic
