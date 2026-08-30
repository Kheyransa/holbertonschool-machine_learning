#!/usr/bin/env python3
"""Module that performs K-means on a dataset using sklearn"""
import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means on a dataset using sklearn

    X: numpy.ndarray of shape (n, d), the dataset
    k: the number of clusters

    Returns: C, clss
        C: numpy.ndarray of shape (k, d), centroid means for each cluster
        clss: numpy.ndarray of shape (n,), cluster index for each point
    """
    model = sklearn.cluster.KMeans(n_clusters=k).fit(X)

    C = model.cluster_centers_
    clss = model.labels_

    return C, clss
