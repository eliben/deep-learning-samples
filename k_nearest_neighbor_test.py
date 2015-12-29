import numpy as np

import k_nearest_neighbor

knn = k_nearest_neighbor.KNearestNeighbor()

train = np.arange(180).reshape(6, -1)
test = np.arange(120).reshape(4, -1)

knn.train(train, None)

dtwo = knn.compute_distances_two_loops(test)
dnone = knn.compute_distances_no_loops(test)

print np.linalg.norm(dtwo - dnone, ord='fro')
