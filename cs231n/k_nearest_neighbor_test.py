import numpy as np
import sys, unittest
from k_nearest_neighbor import KNearestNeighbor

class TestKNearestNeighborDistance(unittest.TestCase):
    """Tests that distance computations all return the same result."""
    def test_arange(self):
        train = np.arange(150).reshape(5, -1)
        test = np.square(np.arange(2, 122)).reshape(4, -1)
        knn = KNearestNeighbor()
        knn.train(train, None)
        d_two = knn.compute_distances_two_loops(test)
        d_one = knn.compute_distances_one_loop(test)
        d_no = knn.compute_distances_no_loops(test)
        self.assertAlmostEqual(0, np.linalg.norm(d_two - d_one, ord='fro'))
        self.assertAlmostEqual(0, np.linalg.norm(d_no - d_one, ord='fro'))


if __name__ == '__main__':
    unittest.main()
