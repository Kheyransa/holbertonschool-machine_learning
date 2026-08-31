#!/usr/bin/env python3
"""Computes the policy and the Monte-Carlo policy gradient"""
import numpy as np


def policy(matrix, weight):
    """
    Computes the policy with a weight of a matrix

    matrix: numpy.ndarray containing the state
    weight: numpy.ndarray containing the weight

    Returns: the policy (action probabilities)
    """
    z = matrix.dot(weight)
    exp = np.exp(z - np.max(z))
    return exp / np.sum(exp, axis=1, keepdims=True)


def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient based on a state
    and a weight matrix

    state: matrix representing the current observation of the
           environment
    weight: matrix of random weight

    Returns: the action and the gradient (in this order)
    """
    state = state.reshape(1, -1)
    probs = policy(state, weight)[0]
    action = np.random.choice(len(probs), p=probs)

    dsoftmax = -probs
    dsoftmax[action] += 1

    gradient = np.outer(state, dsoftmax)

    return action, gradient
