import numpy as np
import unittest

from multiple_linear_regression import feature_normalize


class Test(unittest.TestCase):
    def test_feature_normalize(self):
        X = np.array([[1, 2, 3], [9, 4, 4]])
        X_norm, mu, sigma = feature_normalize(X)
        np.testing.assert_array_equal(mu, [5, 3, 3.5])
        np.testing.assert_array_equal(sigma, [4, 1, 0.5])
        self.assertEqual(X_norm[0][0], -1)
        self.assertEqual(X_norm[-1][-1], 1)


if __name__ == '__main__':
    unittest.main()
