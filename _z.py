import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

import cifar10
import timer

dir = 'datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = cifar10.load_CIFAR10(dir)

print 'Training data shape: ', X_train.shape, X_train.dtype
print 'Training labels shape: ', y_train.shape, y_train.dtype
print 'Test data shape: ', X_test.shape, X_test.dtype
print 'Test labels shape: ', y_test.shape, y_test.dtype

# Subsample to save on time/space
num_training = 500
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 100
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows: each item in these arrays is a 3072-element
# vector representing 3 colors per image pixel.
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

print 'Training data shape: ', X_train.shape, X_train.dtype
print 'Test data shape: ', X_test.shape, X_test.dtype

import k_nearest_neighbor
knn = k_nearest_neighbor.KNearestNeighbor()
knn.train(X_train, y_train)
#dists = knn.compute_distances_two_loops(X_test)

with timer.Timer('Computing distances...'):
    dists = knn.compute_distances_two_loops(X_test)
#print dists_one

# To ensure that our vectorized implementation is correct, we make sure that it
# agrees with the naive implementation. There are many ways to decide whether
# two matrices are similar; one of the simplest is the Frobenius norm. In case
# you haven't seen it before, the Frobenius norm of two matrices is the square
# root of the squared sum of differences of all elements; in other words, reshape
# the matrices into vectors and compute the Euclidean distance between them.
dists_none = knn.compute_distances_no_loops(X_test)
difference = np.linalg.norm(dists - dists_none, ord='fro')
print 'Difference was: %f' % (difference, )
if difference < 0.001:
  print 'Good! The distance matrices are the same'
else:
  print 'Uh-oh! The distance matrices are different'

with timer.Timer('Running label prediction...'):
    y_test_pred = knn.predict_labels(dists, k=5)

## Compute and print the fraction of correctly predicted examples
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print 'Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy)
