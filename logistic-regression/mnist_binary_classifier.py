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
import sys

from mnist_dataset import *
from regression_lib import *


def convert_y_to_binary(y, correct_digit):
    """Converts a vector y taken from MNIST data to binary "is it this digit".

    y: array of digits.
    correct_digit: the digit we expect to be "correct"

    Returns array of +1 or -1; +1 where the original y had the "correct" digit,
    and -1 otherwise. The returned array is always a column vector.
    """
    return np.where(y == correct_digit,
                    np.ones_like(y),
                    -1 * np.ones_like(y)).reshape(y.size, 1)


if __name__ == '__main__':
    train, valid, test = get_mnist_data()
    X_train, y_train = train
    X_valid, y_valid = valid

    #display_mnist_image(X_train[2], y_train[2])

    # Normalization here create NaNs: there may be some columns that are always
    # 0?
    #X_train_normalized, mu, sigma = feature_normalize(X_train)
    #X_train_normalized = X_train
    X_train_augmented = augment_1s_column(X_train)
    X_valid_augmented = augment_1s_column(X_valid)

    # Convert y_train to binary "is this a 4", with +1 for a 4, -1 otherwise.
    # Also reshape it into a column vector as regression_lib expects.
    y_train_binary = convert_y_to_binary(y_train, 4)

    y_valid_binary = convert_y_to_binary(y_valid, 4)

    # Note: if we guess by saying "nothing is 4" we get ~90% accuracy
    NSTEPS = 550
    LEARNING_RATE = 0.05
    REG_BETA=0.02

    lossfunc = lambda X, y, theta: hinge_loss(X, y, theta, reg_beta=REG_BETA)
    gi = gradient_descent(X_train_augmented, y_train_binary, lossfunc,
                          nsteps=NSTEPS, learning_rate=LEARNING_RATE)

    #display_mnist_image(X_valid[8878])
    #sys.exit(1)

    for i, (theta, loss) in enumerate(gi):
        if i % 10 == 0 and i > 0:
            yhat = predict_binary(X_train_augmented, theta)
            print('train accuracy =', np.mean(yhat == y_train_binary))

            yhat_valid = predict_binary(X_valid_augmented, theta)
            print('valid accuracy =', np.mean(yhat_valid == y_valid_binary))

        print(i, loss)

    print('After {0} training steps...'.format(NSTEPS))
    yhat_valid = predict_binary(X_valid_augmented, theta)
    print('valid accuracy =', np.mean(yhat_valid == y_valid_binary))

    for i in range(yhat_valid.size):
        if yhat_valid[i][0] != y_valid_binary[i][0]:
            print('@ {0}: predict {1}, actual {2}'.format(
                i, yhat_valid[i][0], y_valid_binary[i][0]))
