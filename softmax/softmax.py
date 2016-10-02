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


def fully_connected_gradient(x, W):
    """Computes the gradient of a fully connected layer w.r.t. the weights.

    x: (N, 1) input
    W: (T, N) weights

    A fully connected layer acting on the input x is: W.dot(x). This function
    computes the full Jacobian matrix of this formula. The W matrix is flattened
    in row-major order to rows of the Jacobian, such that DijFCt is the
    derivative of output t (the t'th row of W.dot(x)) w.r.t. W[i, j].

    Returns D (T, N * T)
    """
    N = x.shape[0]
    T = W.shape[0]
    D = np.zeros((T, N * T))
    for t in range(T):
        for i in range(T):
            for j in range(N):
                # Computing gradient of the t'th output w.r.t. W[i, j]. Its
                # index in the D matrix is: (t, i*N + j)
                # The t'th output only depends on the t'th row in W. Otherwise
                # the derivative is zero. In the t'th row, each weight is
                # multiplied by the respective x.
                if t == i:
                    D[t, i*N + j] = x[j]
    return D


def softmax_layer(x, W):
    """Computes a "softmax layer" for input vector x and weight matrix W.

    A softmax layer is a fully connected layer followed by the softmax function.
    Mathematically it's softmax(W.dot(x)).

    x: (N, 1) input vector with N features.
    W: (T, N) matrix of weights for N features and T output classes.

    Returns s (T, 1) the result of applying softmax to W.dot(x)
    """
    logits = W.dot(x)
    return softmax(logits)


def softmax_layer_gradient(x, W):
    """Computes the gradient of a "softmax layer" for weight matrix W.

    x: (N, 1) input
    W: (T, N) weights

    A fully connected layer acting on the input x is: W.dot(x). This function
    computes the full Jacobian matrix of this formula. The W matrix is flattened
    in row-major order to rows of the Jacobian, such that DijFCt is the
    derivative of output t (the t'th row of W.dot(x)) w.r.t. W[i, j].

    Returns D (T, N * T)
    """
    logits = W.dot(x)
    return softmax_gradient(logits).dot(fully_connected_gradient(x, W))


def softmax_layer_gradient_direct(x, W):
    """Computes the gradient of a "softmax layer" for weight matrix W.

    Arguments and return value exactly the same as for softmax_layer_gradient.
    The difference is that this function computes the Jacobian "directly" by
    assigning each cell in the matrix, rather than explicitly computing the
    matrix multiplication of the two composed Jacobians.
    """
    N = x.shape[0]
    T = W.shape[0]
    S = softmax_layer(x, W)
    D = np.zeros((T, N * T))
    for t in range(T):
        for i in range(T):
            for j in range(N):
                DiSt = S[t, 0] * (np.float32(i == t) - S[i, 0])
                D[t, i*N + j] = DiSt * x[j, 0]
    return D


def cross_entropy_loss(p, y):
    """Cross-entropy loss between predicted and expected probabilities.

    p: vector of predicted probabilities.
    y: vector of expected probabilities. Has to be the same shape as p.

    Returns a scalar.
    """
    assert(p.shape == y.shape)
    return -np.sum(y * np.log(p))


def cross_entropy_loss_gradient(p, y):
    """Gradient of the cross-entropy loss function for p and y.

    p: (T, 1) vector of predicted probabilities.
    y: (T, 1) vector of expected probabilities; must be one-hot -- one and only
              one element of y is 1; the rest are 0.

    Returns a (1, T) Jacobian for this function.
    """
    assert(p.shape == y.shape and p.shape[1] == 1)
    # py is the value of p at the index where y == 1 (one and only one such
    # index is expected for a one-hot y).
    py = p[y == 1]
    assert(py.size == 1)
    # D is zeros everywhere except at the index where y == 1. The final D has
    # to be a row-vector.
    D = np.zeros_like(p)
    D[y == 1] = -1/py.flat[0]
    return D.flatten()


def softmax_cross_entropy_loss_gradient(x, W, y):
    """Computes the gradient of a cross-entropy loss for a softmax layer.

    x: (N, 1) input
    W: (T, N) weights
    y: (T, 1) correct labels (one-hot vector with one element 1.0, others 0.0)

    Returns D (1, N * T)
    """
    p = softmax_layer(x, W)
    return cross_entropy_loss_gradient(p, y).dot(softmax_layer_gradient(x, W))


def softmax_cross_entropy_loss_gradient_direct(x, W, y):
    """Computes the gradient of a cross-entropy loss for a softmax layer.

    Arguments and return value exactly the same as for
    softmax_cross_entropy_loss_gradient. The difference is that this function
    computes the Jacobian "directly" by assigning each cell in the matrix,
    rather than explicitly computing the matrix multiplication of the two
    composed Jacobians.
    """
    N = x.shape[0]
    T = W.shape[0]
    S = softmax_layer(x, W)
    D = np.zeros(N * T)
    yindex = np.argwhere(y == 1)[0, 0]
    for i in range(T):
        for j in range(N):
            D[i*N + j] = (S[i, 0] - np.float32(i == yindex)) * x[j, 0]
    return D


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
