# Softmax function, its gradient, and combination with other layers.
#
# Eli Bendersky (http://eli.thegreenplace.net)
# This code is in the public domain
from __future__ import print_function
import numpy as np


def softmax(z):
    """Computes softmax function.

    z: array of input values.

    Returns an array of outputs with the same shape as z."""
    # For numerical stability: make the maximum of z's to be 0.
    shiftz = z - np.max(z)
    exps = np.exp(shiftz)
    return exps / np.sum(exps)


def softmax_gradient(z):
    """Computes the gradient of the softmax function.

    z: array of input values where the gradient is computed.

    Returns the full Jacobian of S(z): D (N, N) where DjSi is the partial
    derivative of Si w.r.t. input j.
    """
    Sz = softmax(z)
    # -SjSi can be computed using an outer product between Sz and itself. Then
    # we add back Si for the i=j cases by adding a diagonal matrix with the
    # values of Si on its diagonal.
    dz = -np.outer(Sz, Sz) + np.diag(Sz.flatten())
    return dz


def softmax_gradient_simple(z):
    """Unvectorized computation of the gradient of softmax.

    z: (N, 1) column array of input values.

    Returns dz (N, N) the Jacobian matrix of softmax(z) at the given z. dz[i, j]
    is DjSi - the partial derivative of Si w.r.t. input j.
    """
    Sz = softmax(z)
    N = z.shape[0]
    dz = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            dz[i, j] = Sz[i, 0] * (np.float32(i == j) - Sz[j, 0])
    return dz


if __name__ == '__main__':
    #pa = [2945.0, 2945.5]
    pa = np.array([[1000], [2000], [3000]])
    print(softmax(pa))
    print(stablesoftmax(pa))
