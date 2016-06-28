import numpy as np

import cifar10
import timer

dir = 'datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = cifar10.load_CIFAR10(dir)

# Subsample to save on time/space
num_training = 5000
num_validation = 1000
num_test = 1000

# Our validation set will be num_validation points from the original
# training set.
mask = range(num_training, num_training + num_validation)
X_val = X_train[mask]
y_val = y_train[mask]

# Our training set will be the first num_train points from the original
# training set.
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

# We use the first num_test points of the original test set as our
# test set.
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows: each item in these arrays is a 3072-element
# vector representing 3 colors per image pixel.
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

print 'Train data shape: ', X_train.shape
print 'Train labels shape: ', y_train.shape
print 'Validation data shape: ', X_val.shape
print 'Validation labels shape: ', y_val.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape

# Preprocessing: subtract the mean image
# 1. compute the image mean based on the training data
mean_image = np.mean(X_train, axis=0)

def report_mean_image(mimg):
    print mimg.shape, mimg[:10] # print a few of the elements
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4, 4))
    plt.imshow(mimg.reshape((32, 32, 3)).astype('uint8'))
    plt.show()

# 2. subtract the mean image from train and test data
X_train -= mean_image
X_val -= mean_image
X_test -= mean_image

#import k_nearest_neighbor
#knn = k_nearest_neighbor.KNearestNeighbor()
#knn.train(X_train, y_train)

#with timer.Timer('Computing distances...'):
    #dists = knn.compute_distances_no_loops(X_test)

#with timer.Timer('Running label prediction...'):
    #y_test_pred = knn.predict_labels(dists, k=5)

## Compute and print the fraction of correctly predicted examples
#num_correct = np.sum(y_test_pred == y_test)
#accuracy = float(num_correct) / num_test
#print 'Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy)
