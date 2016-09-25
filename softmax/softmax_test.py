# Tests for softmax.
#
# Eli Bendersky (http://eli.thegreenplace.net)
# This code is in the public domain
from __future__ import print_function
import numpy as np
import unittest

from softmax import *


def eval_numerical_gradient(f, x, verbose=False, h=1e-5):
    """A naive implementation of numerical gradient of f at x.

    f: function taking a single array argument and returning a scalar.
    x: array starting point for evaluation.

    Based on http://cs231n.github.io/assignments2016/assignment1/, with a
    bit of cleanup.

    Returns a numerical gradient
    """
    grad = np.zeros_like(x)
    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h
        fxph = f(x) # evalute f(x + h)
        x[ix] = oldval - h
        fxmh = f(x) # evaluate f(x - h)
        x[ix] = oldval # restore

        # compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h)
        if verbose:
            print(ix, grad[ix])
        it.iternext()
    return grad


class TestSoftmaxGradient(unittest.TestCase):
    def checkSoftmaxGradientSimpleVsVec(self, z):
        dz_vec = softmax_gradient(z)
        dz_simple = softmax_gradient_simple(z)
        np.testing.assert_allclose(dz_vec, dz_simple)

    def test_simple_vs_numerical(self):
        z = np.array([
            [0.2],
            [0.9],
            [-0.3],
            [-0.5]])
        grad = softmax_gradient_simple(z)

        for i in range(z.shape[0]):
            # Compute numerical gradient for output Si w.r.t. all inputs
            # j=0...N-1; this computes one row of the jacobian.
            gradnum = eval_numerical_gradient(lambda z: softmax(z)[i, 0], z)
            np.testing.assert_allclose(grad[i, :].flatten(),
                                       gradnum.flatten(),
                                       rtol=1e-4)

    def test_simple_vs_vectorized_small(self):
        z = np.array([
            [0.2],
            [0.9]])
        self.checkSoftmaxGradientSimpleVsVec(z)

    def test_simple_vs_vectorized_larger(self):
        z = np.array([
            [1.2],
            [0],
            [-0.01],
            [2.12],
            [-0.9]])
        self.checkSoftmaxGradientSimpleVsVec(z)

    def test_simple_vs_vectorized_random(self):
        z = np.random.uniform(low=-2.0, high=2.0, size=(100,1))
        self.checkSoftmaxGradientSimpleVsVec(z)


class TestFullyConnectedGradient(unittest.TestCase):
    def test_small(self):
        W = np.array([
            [2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0]])
        x = np.array([
            [-2.0],
            [6.0],
            [1.6]])
        grad = fully_connected_gradient(x, W)

        for t in range(W.shape[0]):
            # This computes the t'th row in the Jacobian
            gradnum = eval_numerical_gradient(lambda W: W.dot(x)[t, 0], W)
            np.testing.assert_allclose(grad[t, :], gradnum.flatten(order='C'))


if __name__ == '__main__':
    unittest.main()
