#!/usr/bin/env python3
"""Module that performs back propagation over a convolutional layer."""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural
    network.

    Args:
        dZ: numpy.ndarray of shape (m, h_new, w_new, c_new) containing
            the partial derivatives with respect to the unactivated
            output of the convolutional layer
                m: the number of examples
                h_new: the height of the output
                w_new: the width of the output
                c_new: the number of channels in the output
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
            containing the output of the previous layer
                h_prev: the height of the previous layer
                w_prev: the width of the previous layer
                c_prev: the number of channels in the previous layer
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new) containing
            the kernels for the convolution
                kh: the filter height
                kw: the filter width
        b: numpy.ndarray of shape (1, 1, 1, c_new) containing the
            biases applied to the convolution
        padding: string that is either 'same' or 'valid', indicating
            the type of padding used
        stride: tuple of (sh, sw) containing the strides for the
            convolution
                sh: the stride for the height
                sw: the stride for the width

    Returns:
        The partial derivatives with respect to the previous layer
        (dA_prev), the kernels (dW), and the biases (db), respectively.
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph, pw = 0, 0

    A_prev_padded = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    dA_prev_padded = np.zeros(A_prev_padded.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(m):
        a_prev_pad = A_prev_padded[i]
        da_prev_pad = dA_prev_padded[i]
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    v_start = h * sh
                    v_end = v_start + kh
                    h_start = w * sw
                    h_end = h_start + kw

                    a_slice = a_prev_pad[v_start:v_end, h_start:h_end, :]

                    da_prev_pad[v_start:v_end, h_start:h_end, :] += (
                        W[:, :, :, c] * dZ[i, h, w, c]
                    )
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]

    if padding == "same":
        dA_prev = dA_prev_padded[:, ph:ph + h_prev, pw:pw + w_prev, :]
    else:
        dA_prev = dA_prev_padded

    return dA_prev, dW, db
