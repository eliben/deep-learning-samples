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


class TestSoftmaxLayerGradient(unittest.TestCase):
    def checkForData(self, x, W, rtol=1e-5, atol=1e-8):
        # Set custom rtol/atol to the ones used by numpy.isclose; by default
        # assert_allclose uses atol=0 which is problematic for values very close
        # to zero.
        grad = softmax_layer_gradient(x, W)
        grad_direct = softmax_layer_gradient_direct(x, W)
        np.testing.assert_allclose(grad, grad_direct, rtol=rtol, atol=atol)

        for t in range(W.shape[0]):
            gradnum = eval_numerical_gradient(
                lambda W: softmax_layer(x, W)[t, 0], W)
            np.testing.assert_allclose(grad[t, :], gradnum.flatten(order='C'),
                                       rtol=rtol, atol=atol)

    def test_small(self):
        W = np.array([
            [2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0]])
        x = np.array([
            [-2.0],
            [2.0],
            [1.6]])
        self.checkForData(x, W)

    def test_bigger(self):
        W = np.array([
            [2.0, 3.0, 4.0, 1.0],
            [1.0, 1.0, 1.0, -0.1],
            [1.1, 2.1, 3.1, 2.3],
            [1.4, -3.3, -1.0, 1.1]])
        x = np.array([
            [-0.1],
            [0.3],
            [-0.2],
            [0.6]])
        self.checkForData(x, W)

    def test_random_big(self):
        np.random.seed(42)
        N = 80
        T = 10
        W = np.random.uniform(low=-2.0, high=2.0, size=(T, N))
        x = np.random.uniform(low=-1.0, high=1.0, size=(N, 1))
        self.checkForData(x, W)


class TestCrossEntropyLossAndGradient(unittest.TestCase):
    def test_cross_entropy_with_onehot_y(self):
        # Simple test of the cross_entropy_loss function with a one-hot y
        # typical in classification tasks.
        T = 5
        p = np.vstack((1.2, 2.1, 3.1, 4.8, 0.75))

        for i in range(T):
            y = np.zeros((T, 1))
            y[i] = 1.0
            xent = cross_entropy_loss(p, y)
            np.testing.assert_allclose(xent, -np.log(p[i]))

            grad = cross_entropy_loss_gradient(p, y)
            gradnum = eval_numerical_gradient(
                lambda z: cross_entropy_loss(z, y), p)
            np.testing.assert_allclose(grad, gradnum.flatten())


class TestSoftmaxCrossEntropyLossGradient(unittest.TestCase):
    def checkForData(self, x, W, y, rtol=1e-5, atol=1e-8):
        # Set custom rtol/atol to the ones used by numpy.isclose; by default
        # assert_allclose uses atol=0 which is problematic for values very close
        # to zero.
        grad = softmax_cross_entropy_loss_gradient(x, W, y)
        grad_direct = softmax_cross_entropy_loss_gradient_direct(x, W, y)
        np.testing.assert_allclose(grad, grad_direct, rtol=rtol, atol=atol)

        gradnum = eval_numerical_gradient(
            lambda W: cross_entropy_loss(softmax_layer(x, W), y), W)
        np.testing.assert_allclose(grad, gradnum.flatten(order='C'),
                                   rtol=rtol, atol=atol)

    def test_small(self):
        W = np.array([
            [2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0]])
        x = np.array([
            [-2.0],
            [2.0],
            [1.6]])
        y0 = np.array([
            [1.0],
            [0.0]])
        y1 = np.array([
            [0.0],
            [1.0]])
        self.checkForData(x, W, y0)
        self.checkForData(x, W, y1)

    def test_random_big(self):
        np.random.seed(42)
        N = 30
        T = 10
        W = np.random.uniform(low=-2.0, high=2.0, size=(T, N))
        x = np.random.uniform(low=-1.0, high=1.0, size=(N, 1))

        for i in range(T):
            y = np.zeros((T, 1))
            y[i] = 1.0
            self.checkForData(x, W, y)


if __name__ == '__main__':
    unittest.main()
