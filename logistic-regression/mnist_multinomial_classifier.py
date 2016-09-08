from __future__ import print_function
import argparse
import numpy as np
import sys

from mnist_dataset import *
from regression_lib import *


def train_for_digit(X, y, digit, nsteps, learning_rate=0.08, reg_beta=0.02):
    y_binary = convert_y_to_binary(y, digit)

    lossfunc = lambda X, y, theta: cross_entropy_loss_binary(
        X, y, theta, reg_beta=reg_beta)

    gi = gradient_descent(X,
                          y_binary,
                          lossfunc=lossfunc,
                          batch_size=256,
                          nsteps=nsteps,
                          learning_rate=learning_rate)
    # Run GD to completion.
    for i, (theta, _) in enumerate(gi):
        if i % 100 == 0 and i > 0:
            print('{0}...'.format(i), end='')
            sys.stdout.flush()
    print('')
    return theta


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--normalize', action='store_true', default=False,
                           help='Normalize data: (x-mu)/sigma.')
    argparser.add_argument('--nsteps', default=150, type=int,
                           help='Number of steps for gradient descent.')
    argparser.add_argument('--set-seed', default=-1, type=int,
                           help='Set random seed to this number (if > 0).')
    args = argparser.parse_args()

    if args.set_seed > 0:
        np.random.seed(args.set_seed)

    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_mnist_data()

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

    thetas = []
    for digit in range(2):
        print('Training for digit {0}...'.format(digit))
        thetas.append(train_for_digit(X_train_augmented,
                                      y_train,
                                      digit=digit,
                                      nsteps=args.nsteps))

    probs = [predict_logistic_probability(X_test_augmented, theta)
             for theta in thetas]
    print([p.shape for p in probs])
    for i in range(50):
        print('real={0}'.format(y_test[i]))
        print('probs={0}'.format(' '.join(str(p[i, 0]) for p in probs)))
