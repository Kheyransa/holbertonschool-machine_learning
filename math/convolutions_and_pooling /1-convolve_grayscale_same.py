#!/usr/bin/env python3
"""Same Convolution"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images

    images is a numpy.ndarray with shape (m, h, w) containing multiple
    grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw) containing the kernel
    for the convolution
        kh is the height of the kernel
        kw is the width of the kernel

    if necessary, the image is padded with 0's

    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    pad_h = kh // 2
    pad_w = kw // 2

    padded_images = np.pad(
        images,
        pad_width=((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode='constant',
        constant_values=0
    )

    convolved = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            image_slice = padded_images[:, i:i + kh, j:j + kw]
            product = image_slice * kernel
            convolved[:, i, j] = np.sum(product, axis=(1, 2))

    return convolved
