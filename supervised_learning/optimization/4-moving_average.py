#!/usr/bin/env python3
"""Calculates the weighted moving average of a data set"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set

    data: list of data to calculate the moving average of
    beta: weight used for the moving average

    Your moving average calculation should use bias correction

    Returns: a list containing the moving averages of data
    """
    moving_averages = []
    v = 0

    for i, x in enumerate(data):
        v = beta * v + (1 - beta) * x
        v_corrected = v / (1 - beta ** (i + 1))
        moving_averages.append(v_corrected)

    return moving_averages
