#!/usr/bin/env python3
"""Module that performs agglomerative clustering on a dataset"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Performs agglomerative clustering on a dataset with Ward linkage

    X: numpy.ndarray of shape (n, d), the dataset
    dist: the maximum cophenetic distance for all clusters

    Displays the dendrogram with each cluster in a different color

    Returns: clss, numpy.ndarray of shape (n,), cluster indices per point
    """
    Z = scipy.cluster.hierarchy.linkage(X, method='ward')

    clss = scipy.cluster.hierarchy.fcluster(Z, t=dist, criterion='distance')

    scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist)
    plt.show()

    return clss
