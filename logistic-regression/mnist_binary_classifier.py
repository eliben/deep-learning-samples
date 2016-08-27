# A binary linear classifier for MNIST digits.
#
# Poses a binary classification problem - is this image showing digit D (for
# some D, for example "4"); trains a linear classifier to solve the problem.
#
# Eli Bendersky (http://eli.thegreenplace.net)
# This code is in the public domain

from __future__ import print_function
import argparse
import matplotlib.pyplot as plt
import numpy as np

from mnist_dataset import *
from regression_lib import *


if __name__ == '__main__':
    train, valid, test = get_mnist_data()
    X_train, y_train = train

    #display_mnist_image(X_train[2], y_train[2])

    #X_train_normalized, mu, sigma = feature_normalize(X_train)
    X_train_normalized = X_train
    X_train_augmented = np.hstack((np.ones((X_train.shape[0], 1)),
                                           X_train_normalized))

    # Convert y_train to binary "is this a 4", with +1 for a 4, -1 otherwise.
    # Also reshape it into a column vector as regression_lib expects.
    y_train_binary = np.where(y_train == 4,
                              np.ones_like(y_train),
                              -1 * np.ones_like(y_train)).reshape(-1, 1)

    lossfunc = lambda X, y, theta: square_loss(X, y, theta, reg_beta=0.01)
    gi = gradient_descent(X_train_augmented, y_train_binary, lossfunc,
                          nsteps=40, learning_rate=0.01)

    for i, (theta, loss) in enumerate(gi):
        if i % 10 == 0 and i > 0:
            yhat = predict_binary(X_train_augmented, theta)
            print('train accuracy =', np.mean(yhat == y_train_binary))
        print(i, loss)
