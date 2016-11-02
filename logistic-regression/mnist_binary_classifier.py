# A binary linear classifier for MNIST digits.
#
# Poses a binary classification problem - is this image showing digit D (for
# some D, for example "4"); trains a linear classifier to solve the problem.
#
# Eli Bendersky (http://eli.thegreenplace.net)
# This code is in the public domain
from __future__ import print_function
import argparse
import numpy as np
import sys

from mnist_dataset import *
from regression_lib import *


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--type',
                           choices=['binary', 'logistic'],
                           default='logistic',
                           help='Type of classification: binary (yes/no result)'
                                'or logistic (probability of "yes" result).')
    argparser.add_argument('--set-seed', default=-1, type=int,
                           help='Set random seed to this number (if > 0).')
    argparser.add_argument('--nsteps', default=150, type=int,
                           help='Number of steps for gradient descent.')
    argparser.add_argument('--recognize-digit', default=4, type=int,
                           help='Digit to recognize in training.')
    argparser.add_argument('--display-test', default=-1, type=int,
                           help='Display this image from the test data '
                                'set and exit.')
    argparser.add_argument('--normalize', action='store_true', default=False,
                           help='Normalize data: (x-mu)/sigma.')
    argparser.add_argument('--report-mistakes', action='store_true',
                           default=False,
                           help='Report all mistakes made in classification.')
    args = argparser.parse_args()

    if args.set_seed > 0:
        np.random.seed(args.set_seed)

    # Load MNIST data into memory; this may download the MNIST dataset from
    # the web if not already on disk.
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_mnist_data()

    if args.display_test > -1:
        display_mnist_image(X_test[args.display_test],
                            y_test[args.display_test])
        sys.exit(1)

    if args.normalize:
        print('Normalizing data...')
        X_train_normalized, mu, sigma = feature_normalize(X_train)
        X_train_augmented = augment_1s_column(X_train_normalized)
        X_valid_augmented = augment_1s_column((X_valid - mu) / sigma)
        X_test_augmented = augment_1s_column((X_test - mu) / sigma)
    else:
        X_train_augmented = augment_1s_column(X_train)
        X_valid_augmented = augment_1s_column(X_valid)
        X_test_augmented = augment_1s_column(X_test)

    # Convert y_train to binary "is this a the digit D", with +1 for D, -1
    # otherwise. Also reshape it into a column vector as regression_lib expects.
    D = args.recognize_digit
    print('Training for digit', D)
    y_train_binary = convert_y_to_binary(y_train, D)
    y_valid_binary = convert_y_to_binary(y_valid, D)
    y_test_binary = convert_y_to_binary(y_test, D)

    # Hyperparameters.
    LEARNING_RATE = 0.08
    REG_BETA=0.03

    if args.type == 'binary':
        print('Training binary classifier with hinge loss...')
        lossfunc = lambda X, y, theta: hinge_loss(X, y,
                                                  theta, reg_beta=REG_BETA)
    else:
        print('Training logistic classifier with cross-entropy loss...')
        lossfunc = lambda X, y, theta: cross_entropy_loss_binary(
            X, y, theta, reg_beta=REG_BETA)
    n = X_train_augmented.shape[1]
    gi = gradient_descent(X_train_augmented,
                          y_train_binary,
                          init_theta=np.random.randn(n, 1),
                          lossfunc=lossfunc,
                          batch_size=256,
                          nsteps=args.nsteps,
                          learning_rate=LEARNING_RATE)

    for i, (theta, loss) in enumerate(gi):
        if i % 50 == 0 and i > 0:
            print(i, loss)
            # We use predict_binary for both binary and logistic classification.
            # See comment on predict_binary to understand why it works for
            # logistic as well.
            yhat = predict_binary(X_train_augmented, theta)
            yhat_valid = predict_binary(X_valid_augmented, theta)
            print('train accuracy =', np.mean(yhat == y_train_binary))
            print('valid accuracy =', np.mean(yhat_valid == y_valid_binary))

    print('After {0} training steps...'.format(args.nsteps))
    print('loss =', loss)
    yhat_valid = predict_binary(X_valid_augmented, theta)
    yhat_test = predict_binary(X_test_augmented, theta)
    print('valid accuracy =', np.mean(yhat_valid == y_valid_binary))
    print('test accuracy =', np.mean(yhat_test == y_test_binary))

    # For logistic, get predicted probabilities as well.
    if args.type == 'logistic':
        yhat_test_prob = predict_logistic_probability(X_test_augmented, theta)

    if args.report_mistakes:
        for i in range(yhat_test.size):
            if yhat_test[i][0] != y_test_binary[i][0]:
                print('@ {0}: predict {1}, actual {2}'.format(
                    i, yhat_test[i][0], y_test_binary[i][0]), end='')
                if args.type == 'logistic':
                    print('; prob={0}'.format(yhat_test_prob[i][0]))
                else:
                    print('')
