# A multiclass logistic regression using softmax for MNIST digits.
#
# Eli Bendersky (http://eli.thegreenplace.net)
# This code is in the public domain
from __future__ import print_function
import argparse
import numpy as np
import sys

from mnist_dataset import *
from regression_lib import *


def train(X, y, nsteps, learning_rate=0.09, reg_beta=0.01):
    """Train a logistic regression binary classifier for recognizing the digit.
    """
    k, n = X.shape
    assert y.shape == (k,)

    lossfunc = lambda X, y, W: softmax_cross_entropy_loss(
        X, y, W, reg_beta=reg_beta)

    init_W = np.random.randn(n, 10)

    # The gradient_descent is generic across different algorithms, so it uses
    # the name "theta" for the regression parameters. Here we assign our W into
    # theta.
    gi = gradient_descent(X,
                          y,
                          init_theta=init_W,
                          lossfunc=lossfunc,
                          batch_size=256,
                          nsteps=nsteps,
                          learning_rate=learning_rate)
    # Run GD to completion.
    for i, (W, loss) in enumerate(gi):
        if i % 100 == 0 and i > 0:
            print('{0}... [loss={1}]'.format(i, loss))
            sys.stdout.flush()
    print('')
    return W


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--normalize', action='store_true', default=False,
                           help='Normalize data: (x-mu)/sigma.')
    argparser.add_argument('--nsteps', default=150, type=int,
                           help='Number of steps for gradient descent.')
    argparser.add_argument('--set-seed', default=-1, type=int,
                           help='Set random seed to this number (if > 0).')
    argparser.add_argument('--load-weights', type=str,
                           metavar='filename',
                           help='Load trained weights from this pickle file '
                                'instead of training.')
    argparser.add_argument('--save-weights', type=str,
                           metavar='filename',
                           help='Save trained weights to this pickle file. '
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

    # Here "weights" is a matrix with N rows and 10 columns, where N is the
    # number of features (pixels) in every MNIST image.
    if args.load_weights:
        print('Loading weights from "{0}"'.format(args.load_weights))
        with open(args.load_weights, 'rb') as f:
            W = pickle.load(f)
    else:
        # Train a softmax classifier for every image [0..9]; W is the trained
        # weights.
        W = train(X_train_augmented, y_train, nsteps=args.nsteps)
    print('W shape:', W.shape)

    if args.save_weights:
        print('Saving weights to "{0}"'.format(args.save_weights))
        with open(args.save_weights, 'wb') as f:
            pickle.dump(W, f)

    probs = softmax_layer(X_test_augmented, W)
    print('Probs shape:', probs.shape)

    # Softmax assigns probabilities of each digits per data item; argmax will
    # pinpoint the column with the highest probability.
    predictions = np.argmax(probs, axis=1)
    print('Predictions shape:', predictions.shape)
    print('y_test shape:', y_test.shape)
    print('test accuracy =', np.mean(predictions == y_test))

    if args.report_mistakes:
        for i in range(y_test.size):
            if y_test[i] != predictions[i]:
                print('{0}: real={1} pred={2}'.format(i, y_test[i],
                                                      predictions[i]))
                print('  probs=', ' '.join('{0:.2f}'.format(p)
                                           for p in probs[i, :]))
