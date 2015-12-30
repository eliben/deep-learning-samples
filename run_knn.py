# Runs a KNN classifier on a subset of CIFAR-10 data.
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
num_training = 50000
X_train = X_train[:num_training]
y_train = y_train[:num_training]

num_test = 100
X_test = X_test[:num_test]
y_test = y_test[:num_test]

# Reshape the image data into rows: each item in these arrays is a 3072-element
# vector representing 3 colors per image pixel.
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

print 'Reshaped training data shape: ', X_train.shape, X_train.dtype
print 'Reshaped test data shape: ', X_test.shape, X_test.dtype

import k_nearest_neighbor
knn = k_nearest_neighbor.KNearestNeighbor()
knn.train(X_train, y_train)

with timer.Timer('Computing distances'):
    dists = knn.compute_distances_no_loops(X_test)

with timer.Timer('Running label prediction'):
    y_test_pred = knn.predict_labels(dists, k=5)

# Compute and print the fraction of correctly predicted examples
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print 'Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy)
