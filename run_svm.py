import numpy as np

import cifar10
import linear_svm
import softmax
import math_utils
import timer

dir = 'datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = cifar10.load_CIFAR10(dir)

# Subsample to save on time/space
num_training = 5000
num_validation = 100
num_test = 100

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

# 3. append the bias dimension of ones (i.e. bias trick) so that our SVM
# only has to worry about optimizing a single weight matrix W.
# Also, transform data matrices so that each image is a column.
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))]).T
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))]).T
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))]).T

print X_train.shape, X_val.shape, X_test.shape

W = np.random.randn(10, 3073) * 0.0001

with timer.Timer('SVM loss naive'):
    loss, grad = linear_svm.svm_loss_naive(W, X_train, y_train, 0.00001)
with timer.Timer('SVM loss vectorized'):
    loss, grad = linear_svm.svm_loss_vectorized(W, X_train, y_train, 0.00001)

classifier = linear_svm.LinearSVM()

# Note: the softmax classifier works but it's slow (I only have the
# non-vectorized version implemented so far). Therefore it remains commented out
# by default.
#classifier = softmax.Softmax()

loss_hist = classifier.train(X_train, y_train, learning_rate=1e-7, reg=5e4,
                       num_iters=800, verbose=True)

y_train_pred = classifier.predict(X_train)
print 'training accuracy: %f' % (np.mean(y_train == y_train_pred), )
y_val_pred = classifier.predict(X_val)
print 'validation accuracy: %f' % (np.mean(y_val == y_val_pred), )
