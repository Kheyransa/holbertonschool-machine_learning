#!/usr/bin/env python3
"""Module that performs forward propagation over a pooling layer."""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural
    network.

    Args:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
            containing the output of the previous layer
                m: the number of examples
                h_prev: the height of the previous layer
                w_prev: the width of the previous layer
                c_prev: the number of channels in the previous layer
        kernel_shape: tuple of (kh, kw) containing the size of the
            kernel for the pooling
                kh: the kernel height
                kw: the kernel width
        stride: tuple of (sh, sw) containing the strides for the
            pooling
                sh: the stride for the height
                sw: the stride for the width
        mode: string containing either 'max' or 'avg', indicating
            whether to perform maximum or average pooling,
            respectively

    Returns:
        The output of the pooling layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_new = (h_prev - kh) // sh + 1
    w_new = (w_prev - kw) // sw + 1

    A = np.zeros((m, h_new, w_new, c_prev))

    for i in range(h_new):
        for j in range(w_new):
            v_start = i * sh
            v_end = v_start + kh
            h_start = j * sw
            h_end = h_start + kw
            A_slice = A_prev[:, v_start:v_end, h_start:h_end, :]

            if mode == 'max':
                A[:, i, j, :] = np.max(A_slice, axis=(1, 2))
            else:
                A[:, i, j, :] = np.mean(A_slice, axis=(1, 2))

    return A
