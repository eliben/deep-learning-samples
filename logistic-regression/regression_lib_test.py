from __future__ import print_function
import numpy as np
import unittest

from regression_lib import (
    hinge_loss, square_loss, predict_logistic_probability,
    cross_entropy_loss_binary)


def hinge_loss_simple(X, y, theta, reg_beta=0.0):
    """Unvectorized version of hinge loss.

    Closely follows the formulae without vectorizing optimizations, so it's
    easier to understand and correlate to the math.
    """
    k, n = X.shape
    loss = 0
    dtheta = np.zeros_like(theta)
    for i in range(k):
        # The contribution of each data item.
        x_i = X[i, :]
        y_i = y[i, 0]
        m_i = x_i.dot(theta).flat[0] * y_i  # margin for i
        loss += np.maximum(0, 1 - m_i) / k
        for j in range(n):
            # This data item contributes gradients to each of the theta
            # components.
            dtheta[j, 0] += -y_i * x_i[j] / k if m_i < 1 else 0

    # Add regularization.
    loss += np.dot(theta.T, theta) * reg_beta / 2
    for j in range(n):
        dtheta[j, 0] += reg_beta * theta[j, 0]

    return loss, dtheta


def cross_entropy_loss_binary_simple(X, y, theta):
    """Unvectorized cross-entropy loss for binary classification."""
    k, n = X.shape
    yhat_prob = predict_logistic_probability(X, theta)
    loss = np.mean(np.where(y == 1,
                            -np.log(yhat_prob),
                            -np.log(1 - yhat_prob)))

    dtheta = np.zeros_like(theta)
    for i in range(k):
        for j in range(n):
            if y[i] == 1:
                dtheta[j, 0] += (yhat_prob[i, 0] - 1 ) * X[i, j]
            else:
                dtheta[j, 0] += yhat_prob[i, 0] * X[i, j]
    return loss, dtheta / k


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


class TestSquareLoss(unittest.TestCase):
    def test_simple_vs_numerical_noreg(self):
        X = np.array([
            [0.1, 0.2, -0.3],
            [0.6, -0.5, 0.1],
            [0.6, -0.4, 0.3],
            [-0.2, 0.4, 2.2]])
        theta = np.array([
            [0.2],
            [-1.5],
            [2.35]])
        y = np.array([
            [1],
            [-1],
            [1],
            [1]])

        loss, grad = square_loss(X, y, theta)
        gradnum = eval_numerical_gradient(
            lambda theta: square_loss(X, y, theta)[0], theta, h=1e-8)
        np.testing.assert_allclose(grad, gradnum, rtol=1e-4)

    def test_simple_vs_numerical_withreg(self):
        # Same test with a regularization factor.
        X = np.array([
                [0.1, 0.2, -0.3],
                [0.6, -0.5, 0.1],
                [0.6, -0.4, 0.3],
                [-0.2, 0.4, 2.2]])
        theta = np.array([
            [0.2],
            [-1.5],
            [2.35]])
        y = np.array([
            [1],
            [-1],
            [1],
            [1]])

        beta = 0.1
        loss, grad = square_loss(X, y, theta, reg_beta=beta)
        gradnum = eval_numerical_gradient(
            lambda theta: square_loss(X, y, theta, reg_beta=beta)[0],
            theta, h=1e-8)
        np.testing.assert_allclose(grad, gradnum, rtol=1e-4)


class TestHingeLoss(unittest.TestCase):
    def checkHingeLossSimpleVsVec(self, X, y, theta, reg_beta=0.0):
        loss_vec, dtheta_vec = hinge_loss(X, y, theta, reg_beta)
        loss_simple, dtheta_simple = hinge_loss_simple(X, y, theta, reg_beta)
        self.assertAlmostEqual(loss_vec, loss_simple)
        np.testing.assert_allclose(dtheta_vec, dtheta_simple)

    def test_hinge_loss_small(self):
        X = np.array([
                [0.1, 0.2, -0.3],
                [0.6, -0.5, 0.1],
                [0.6, -0.4, 0.3],
                [-0.2, 0.4, 2.2]])
        theta = np.array([
            [0.2],
            [-1.5],
            [2.35]])
        y = np.array([
            [1],
            [-1],
            [1],
            [1]])
        # Without regularization.
        self.checkHingeLossSimpleVsVec(X, y, theta, reg_beta=0.0)

        # With regularization.
        beta = 0.05
        self.checkHingeLossSimpleVsVec(X, y, theta, reg_beta=beta)

        # With regularization, compare to numerical gradient.
        loss, grad = hinge_loss(X, y, theta, reg_beta=beta)
        gradnum = eval_numerical_gradient(
            lambda theta: hinge_loss(X, y, theta, reg_beta=beta)[0],
            theta, h=1e-8)
        np.testing.assert_allclose(grad, gradnum, rtol=1e-4)

    def test_hinge_loss_larger_random(self):
         np.random.seed(1)
         k, n = 20, 5
         X = np.random.uniform(low=0, high=1, size=(k,n))
         theta = np.random.randn(n, 1)
         y = np.random.choice([-1, 1], size=(k,1))
         self.checkHingeLossSimpleVsVec(X, y, theta)

    def test_hinge_loss_even_larger_random(self):
         np.random.seed(1)
         k, n = 350, 15
         X = np.random.uniform(low=0, high=1, size=(k,n))
         theta = np.random.randn(n, 1) * 2
         y = np.random.choice([-1, 1], size=(k,1))
         self.checkHingeLossSimpleVsVec(X, y, theta)


class TestCrossEntropyBinaryLoss(unittest.TestCase):
    def checkXentLossSimpleVsVec(self, X, y, theta, reg_beta=0.0):
        loss_vec, dtheta_vec = cross_entropy_loss_binary(X, y, theta)
        loss_simple, dtheta_simple = cross_entropy_loss_binary_simple(
            X, y, theta)
        self.assertAlmostEqual(loss_vec, loss_simple)
        np.testing.assert_allclose(dtheta_vec, dtheta_simple)

    def test_xent_loss_oneitem(self):
        X = np.array([[0.1, 0.2, -0.3]])
        theta = np.array([
            [0.2],
            [-1.5],
            [2.35]])
        y = np.array([[1]])

        self.checkXentLossSimpleVsVec(X, y, theta)

        loss, grad = cross_entropy_loss_binary_simple(X, y, theta)
        gradnum = eval_numerical_gradient(
            lambda theta: cross_entropy_loss_binary_simple(X, y, theta)[0],
            theta, h=1e-8)
        np.testing.assert_allclose(grad, gradnum, rtol=1e-4)

    def test_xent_loss_small(self):
        X = np.array([
                [0.1, 0.2, -0.3],
                [0.6, -0.5, 0.1],
                [0.6, -0.4, 0.3],
                [-0.2, 0.4, 2.2]])
        theta = np.array([
            [0.2],
            [-1.5],
            [2.35]])
        y = np.array([
            [1],
            [-1],
            [1],
            [1]])

        self.checkXentLossSimpleVsVec(X, y, theta)

        loss, grad = cross_entropy_loss_binary_simple(X, y, theta)
        gradnum = eval_numerical_gradient(
            lambda theta: cross_entropy_loss_binary_simple(X, y, theta)[0],
            theta, h=1e-8)
        np.testing.assert_allclose(grad, gradnum, rtol=1e-4)


class TestPredictLogisticProbability(unittest.TestCase):
    def test_close_to_zero(self):
        # For very large negative z, predicted probability is close to zero.
        X = np.array([
            [10.0, 20.0],
            [20.0, 30.0],
            [30.0, 40.0]])
        theta = np.array([[-5], [-6]])
        p = predict_logistic_probability(X, theta)
        np.testing.assert_allclose(p, np.zeros_like(p), atol=1e-8)

    def test_close_to_one(self):
        # For very large positive z, predicted probability is close to one.
        X = np.array([
            [10.0, 20.0],
            [20.0, 30.0],
            [30.0, 40.0]])
        theta = np.array([[3], [4]])
        p = predict_logistic_probability(X, theta)
        np.testing.assert_allclose(p, np.ones_like(p), atol=1e-8)

    def test_half(self):
        # For z=0 we get probability 0.5
        X = np.array([
            [10.0, 20.0],
            [20.0, 40.0],
            [40.0, 80.0]])
        theta = np.array([[-2], [1]])
        p = predict_logistic_probability(X, theta)
        np.testing.assert_allclose(p, np.full(p.shape, 0.5))


if __name__ == '__main__':
    unittest.main()
