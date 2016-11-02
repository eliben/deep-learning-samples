# A multiclass logistic regression (OvA) for MNIST digits.
#
# Trains 10 different logistic regressions, one for each digit, and classifies
# new inputs based on the highest probability among all the trained classifiers.
#
# Eli Bendersky (http://eli.thegreenplace.net)
# This code is in the public domain
from __future__ import print_function
import argparse
import numpy as np
import sys

from mnist_dataset import *
from regression_lib import *


def train_for_digit(X, y, digit, nsteps, learning_rate=0.12, reg_beta=0.02):
    """Train a logistic regression binary classifier for recognizing the digit.
    """
    y_binary = convert_y_to_binary(y, digit)

    lossfunc = lambda X, y, theta: cross_entropy_loss_binary(
        X, y, theta, reg_beta=reg_beta)

    n = X.shape[1]
    gi = gradient_descent(X,
                          y_binary,
                          init_theta=np.random.randn(n, 1),
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
    argparser.add_argument('--load-thetas', type=str,
                           metavar='filename',
                           help='Load trained thetas from this pickle file '
                                'instead of training.')
    argparser.add_argument('--save-thetas', type=str,
                           metavar='filename',
                           help='Save trained thetas to this pickle file. '
                                'Helpful since training can take a long time.')
    argparser.add_argument('--report-mistakes', action='store_true',
                           default=False,
                           help='Report all mistakes made in classification.')
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

    # Training may take a long time and we may want to experiment with a trained
    # model; therefore the save-thetas and load-thetas arguments let us save the
    # trained model parameters into a pickle file and quickly reload them on a
    # subsequent run.
    if args.load_thetas:
        print('Loading thetas from "{0}"'.format(args.load_thetas))
        with open(args.load_thetas, 'rb') as f:
            thetas = pickle.load(f)
    else:
        # Train a logistic classifier for every image [0..9]; thetas[n] will
        # hold the regression parameters for recognizing digit n.
        thetas = []
        for digit in range(10):
            print('Training for digit {0}...'.format(digit))
            thetas.append(train_for_digit(X_train_augmented,
                                          y_train,
                                          digit=digit,
                                          nsteps=args.nsteps))
    print('thetas shape:', ', '.join([str(theta.shape) for theta in thetas]))

    if args.save_thetas:
        print('Saving thetas to "{0}"'.format(args.save_thetas))
        with open(args.save_thetas, 'wb') as f:
            pickle.dump(thetas, f)

    # Compute probabilities for every digit and stack them into a (k, 10) matrix
    # where allprobs[i, j] is the predicted probability for test sample i being
    # the digit j. Note that these probabilities come from different classifiers
    # so they don't add up to 1.
    probs = [predict_logistic_probability(X_test_augmented, theta)
             for theta in thetas]
    allprobs = np.hstack(probs)
    print(allprobs.shape)

    # Use argmax to find the highest probability for every row.
    predictions = np.argmax(allprobs, axis=1)
    print(predictions.shape)
    print(y_test.shape)

    print('test accuracy =', np.mean(predictions == y_test))
    if args.report_mistakes:
        for i in range(y_test.size):
            if y_test[i] != predictions[i]:
                print('{0}: real={1} pred={2}'.format(i, y_test[i],
                                                      predictions[i]))
                print('  probs=', ' '.join('{0:.2f}'.format(p)
                                           for p in allprobs[i, :]))
