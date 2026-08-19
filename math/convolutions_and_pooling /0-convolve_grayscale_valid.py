#!/usr/bin/env python3
"""Valid Convolution"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images

    images is a numpy.ndarray with shape (m, h, w) containing multiple
    grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw) containing the kernel
    for the convolution
        kh is the height of the kernel
        kw is the width of the kernel

    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    out_h = h - kh + 1
    out_w = w - kw + 1

    convolved = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            image_slice = images[:, i:i + kh, j:j + kw]
            product = image_slice * kernel
            convolved[:, i, j] = np.sum(product, axis=(1, 2))

    return convolved
