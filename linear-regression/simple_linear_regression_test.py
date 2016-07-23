import numpy as np
import unittest

from simple_linear_regression import compute_cost


class Test(unittest.TestCase):
    def test_compute_cost(self):
        self.assertAlmostEqual(
            compute_cost(
                np.column_stack(([1, 2, 3], )),
                np.column_stack(([7, 3, 5], )),
                m=2,
                b=3),
            12.0)


if __name__ == '__main__':
    unittest.main()
