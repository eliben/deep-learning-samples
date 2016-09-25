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

    z: (T, 1) array of input values where the gradient is computed. T is the
       number of output classes.

    Returns D (T, T) the Jacobian matrix of softmax(z) at the given z. D[i, j]
    is DjSi - the partial derivative of Si w.r.t. input j.
    """
    Sz = softmax(z)
    # -SjSi can be computed using an outer product between Sz and itself. Then
    # we add back Si for the i=j cases by adding a diagonal matrix with the
    # values of Si on its diagonal.
    D = -np.outer(Sz, Sz) + np.diag(Sz.flatten())
    return D


def softmax_gradient_simple(z):
    """Unvectorized computation of the gradient of softmax.

    z: (T, 1) column array of input values.

    Returns D (T, T) the Jacobian matrix of softmax(z) at the given z. D[i, j]
    is DjSi - the partial derivative of Si w.r.t. input j.
    """
    Sz = softmax(z)
    N = z.shape[0]
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            D[i, j] = Sz[i, 0] * (np.float32(i == j) - Sz[j, 0])
    return D


def softmax_layer(x, W):
    """Computes a "softmax layer" for input vector x and weight matrix W.
    
    A softmax layer is a fully connected layer followed by the softmax function.
    Mathematically it's softmax(W.dot(x)).

    x: (N, 1) input vector with N features.
    W: (T, N) matrix of weights for N features and T output classes.

    Returns s (T, 1) the result of applying softmax to W.dot(x)
    """
    N = x.shape[0]
    T = W.shape[0]
    assert W.shape[1] == N
    assert x.shape[1] == 1
    logits = W.dot(x)
    return softmax(logits)


def fully_connected_gradient(x, W):
    N = x.shape[0]
    T = W.shape[0]
    
    


if __name__ == '__main__':
    #pa = [2945.0, 2945.5]
    #pa = np.array([[1000], [2000], [3000]])
    #print(softmax(pa))
    #print(stablesoftmax(pa))

    W = np.array([
        [2, 3, 4],
        [3, 5, -1]])
    x = np.array([
        [0.1],
        [-0.2],
        [0.3]])
    print(softmax_layer(x, W))
