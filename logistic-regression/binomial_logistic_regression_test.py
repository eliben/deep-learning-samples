import numpy as np
import unittest

from binomial_logistic_regression import hinge_loss


# The idea is the simplest implementation of hinge loss with explicit loops,
# so we can easily compare more complex implementations to.
def hinge_loss_simple(X, y, theta):
    """Unvectorized version of hinge loss."""
    k, n = X.shape
    loss = 0
    dtheta = np.zeros_like(theta)
    for i in range(k):
        x_i = X[i, :]
        y_i = y[i, 0]
        # i'th margin
        m_i = x_i.dot(theta).flat[0] * y_i
        loss += np.maximum(m_i, 1 - m_i)
        for j in range(n):
            dtheta[j, 0] += -y_i * x_i[j] if m_i < 1 else 0
    return loss, dtheta



class Test(unittest.TestCase):
    def test_hinge_loss(self):
        X = np.array([
                [0.1, 0.2, -0.3],
                [0.6, -0.5, 0.1],
                [0.6, -0.4, 0.3],
                [-0.2, 0.4, 0.2]])
        theta = np.array([
            [0.1],
            [-0.5],
            [1]])
        y = np.array([
            [1],
            [-1],
            [1],
            [1]])
        loss, dtheta = hinge_loss_simple(X, y, theta)
        print(loss)
        print(dtheta)


        #self.assertAlmostEqual(
            #compute_cost(
                #np.column_stack(([1, 2, 3], )),
                #np.column_stack(([7, 3, 5], )),
                #m=2,
                #b=3),
            #12.0)


if __name__ == '__main__':
    unittest.main()
