# Simple binary linear classifier with synthetic data.
#
# Eli Bendersky (http://eli.thegreenplace.net)
# This code is in the public domain
from __future__ import print_function
import argparse
import matplotlib.pyplot as plt
import numpy as np
from timer import Timer
import sys

from regression_lib import *


def generate_data(k, num_neg_outliers=0):
    """Generates k data items with correct labels (+1 or -1) for each item.

    k: number of data items to generate.
    num_neg_outliers: number of outliers for the negative samples.

    Returns X (k, 2) - k data items in 2D, and y (k, 1) - the correct label
    (+1 or -1) for each data item in X.
    """
    kneg, kpos = k / 2, k / 2
    kneg_regular = kneg - num_neg_outliers
    # Generate positive data items and negative data items; for negatives, the
    # "regulars" are generated using different parameters from "outliers".
    positives = (np.full((kpos, 2), 3.0) +
                 np.random.normal(scale=0.9, size=(kpos, 2)))
    outliers = (np.hstack((np.ones((num_neg_outliers, 1)) * 3,
                           np.ones((num_neg_outliers, 1)) * 5)) +
                np.random.normal(scale=0.8, size=(num_neg_outliers, 2)))
    negatives = (np.full((kneg_regular, 2), 1.0) +
                 np.random.normal(scale=0.7, size=(kneg_regular, 2)))

    # Stack all items into the same array. To match y, first come all the
    # positives then all the negatives.
    X = np.vstack((positives, negatives, outliers))

    # Create labels. We have kpos +1s followed by kneg -1s.
    y = np.vstack((np.full((kpos, 1), 1.0), np.full((kneg, 1), -1.0)))

    # Stack X and y together so we can shuffle them together.
    Xy = np.random.permutation(np.hstack((X, y)))
    return Xy[:, 0:2], Xy[:, 2].reshape(-1, 1)


def plot_data_scatterplot(X, y, thetas=[]):
    """Plots data as a scatterplot, with contour lines for thetas.

    X: (k, 2) data items.
    y: (k, 1) result (+1 or -1) for each data item in X.
    thetas: list of (theta array, label) pairs to plot contours.

    Plots +1 data points as a green x, -1 as red o.
    """
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    pos = [(X[k, 0], X[k, 1]) for k in range(X.shape[0]) if y[k, 0] == 1]
    neg = [(X[k, 0], X[k, 1]) for k in range(X.shape[0]) if y[k, 0] == -1]

    ax.scatter(*zip(*pos), c='darkgreen', marker='x')
    ax.scatter(*zip(*neg), c='red', marker='o', linewidths=0)

    colors = iter(('blue', 'purple', 'black'))
    contours = []
    for theta, _ in thetas:
        xs = np.linspace(-2, 6, 200)
        ys = np.linspace(-2, 6, 200)
        xsgrid, ysgrid = np.meshgrid(xs, ys)
        plane = np.zeros_like(xsgrid)
        for i in range(xsgrid.shape[0]):
            for j in range(xsgrid.shape[1]):
                plane[i, j] = np.array([1, xsgrid[i, j], ysgrid[i, j]]).dot(
                    theta)
        contours.append(ax.contour(xsgrid, ysgrid, plane,
                                   colors=colors.next(), levels=[0]))

    if thetas:
        plt.legend([cs.collections[0] for cs in contours],
                   [label for theta, label in thetas])
    fig.savefig('binary.png', dpi=80)
    plt.show()


def L01_loss(X, y, theta):
    """Computes the L0/1 loss for the data X using theta.

    X: (k, n) k rows of data items, each having n features; augmented.
    y: (k, 1) correct classifications (+1 or -1) for each item.
    theta: (n, 1) regression parameters.

    Returns the total L0/1 loss over the whole data set. The total L0/1 loss
    is the number of mispredicted items (where y doesn't match yhat).
    """
    results = predict_binary(X, theta)
    return np.count_nonzero(results != y)


def search_best_L01_loss(X, y, theta_start=None,
                         npoints_per_t=150, tmargin=0.1):
    """Hacky exhaustive search for the best L0/1 loss for given X and y.

    X: (k, n) data items.
    y: (k, 1) result (+1 or -1) for each data item in X.
    theta_start: (3, 1) theta to start search from.
    npoints_per_t: number of points to search per dimension of theta.
    tmargin: search within [-tmargin, tmargin] of theta_start.

    Since the search is combinatorial, it is slow and works best when we begin
    with a reasonable good theta. When theta is already close to optimal, this
    search will do a good job finding the best theta in its vicinity. A
    realistic approach which I didn't commit to code (but it could be easily
    done) is to run this search on multiple "zoom" levels (kinda like simulated
    annealing).

    Returns a pair best_theta, best_loss.
    """
    if theta_start is None:
        theta_start = np.array([[1], [1], [1]])
    assert theta_start.shape == (3, 1)

    k = X.shape[0]
    best_loss = k
    best_theta = theta_start
    t0_range = np.linspace(theta_start[0, 0] + tmargin,
                           theta_start[0, 0] - tmargin,
                           npoints_per_t)
    t1_range = np.linspace(theta_start[1, 0] + tmargin,
                           theta_start[1, 0] - tmargin,
                           npoints_per_t)
    t2_range = np.linspace(theta_start[2, 0] + tmargin,
                           theta_start[2, 0] - tmargin,
                           npoints_per_t)
    for t0 in t0_range:
        for t1 in t1_range:
            for t2 in t2_range:
                theta = np.array([[t0], [t1], [t2]])
                loss = L01_loss(X, y, theta)
                if loss < best_loss:
                    best_loss = loss
                    best_theta = theta

    return best_theta, best_loss


def run_gradient_descent_search(X, y, lossfunc, max_nsteps, learning_rate,
                                verbose=False):
    """Helper function to run GD search for the given data and loss function.

    For help on arguments, see the gradient_descent function. max_nsteps is like
    nsteps except that this function will stop once the loss isn't changing much
    any more, which may take fewer than max_nsteps steps.
    """
    n = X.shape[1]
    gradient_descent_iter = gradient_descent(X, y,
                                             init_theta=np.random.randn(n, 1),
                                             lossfunc=lossfunc,
                                             nsteps=max_nsteps,
                                             learning_rate=learning_rate)
    # Run GD until the changes in loss converge to some small value, or until
    # max_nstepsis reached.
    prev_loss = sys.float_info.max
    converge_step = 0
    for i, (theta, loss) in enumerate(gradient_descent_iter):
        if verbose:
            print(i, ':', loss)
        # Convergence of loss beneath a small threshold: this threshold can also
        # be made configurable, if needed.
        if abs(loss - prev_loss) < 1e-5:
            converge_step = i
            break
        prev_loss = loss
    print('... loss converged at step {0}'.format(converge_step))
    return theta


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--plot', action='store_true', default=False,
                           help='Produce scatterplot with fit contours')
    argparser.add_argument('--search01', action='store_true', default=False,
                           help='Run combinatorial search for best L0/1 loss')
    argparser.add_argument('--verbose-gd', action='store_true', default=False,
                           help='Verbose output from gradient-descent search')
    argparser.add_argument('--normalize', action='store_true', default=False,
                           help='Normalize data: (x-mu)/sigma.')

    args = argparser.parse_args()

    # For reproducibility
    np.random.seed(42)

    X_train, y_train = generate_data(400, num_neg_outliers=10)
    print('X_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)

    if args.normalize:
        X_train_normalized, mu, sigma = feature_normalize(X_train)
    else:
        X_train_normalized, mu, sigma = X_train, 0, 1

    X_train_augmented = augment_1s_column(X_train)
    print('X_train_augmented shape:', X_train_augmented.shape)

    # A pretty good theta determined by a long run of search_best_L01_loss with
    # coarse granularity. Works for the seed set above. For different data,
    # we'll need to find a new theta for good L01 loss.
    theta = np.array([-0.9647, 0.2545, 0.2416]).reshape(-1, 1)
    print('Initial theta:\n', theta)
    print('Initial loss:', L01_loss(X_train_augmented, y_train, theta))

    if args.search01:
        with Timer('searching for best L01 loss'):
            best_theta, best_loss = search_best_L01_loss(X_train_augmented,
                                                         y_train,
                                                         theta)
        print('Best theta:\n', best_theta)
        print('Best loss:', best_loss)
    else:
        best_theta, best_loss = theta, L01_loss(X_train_augmented, y_train,
                                                theta)

    # Run GD with square loss.
    square_nsteps = 5000
    square_learning_rate = 0.01
    print('Running GD with square loss for {0} steps, learning_rate={1}'.format(
        square_nsteps, square_learning_rate))
    theta_square = run_gradient_descent_search(
        X_train_augmented,
        y_train,
        lossfunc=square_loss,
        max_nsteps=square_nsteps,
        learning_rate=square_learning_rate,
        verbose=args.verbose_gd)
    print('Found theta:\n', theta_square)
    print('0/1 loss:', L01_loss(X_train_augmented, y_train, theta_square))

    # Run GD with hinge loss.
    hinge_nsteps = 5000
    hinge_learning_rate = 0.01
    print('Running GD with hinge loss for {0} steps, learning_rate={1}'.format(
        hinge_nsteps, hinge_learning_rate))
    theta_hinge = run_gradient_descent_search(
        X_train_augmented,
        y_train,
        lossfunc=hinge_loss,
        max_nsteps=hinge_nsteps,
        learning_rate=hinge_learning_rate,
        verbose=args.verbose_gd)
    print('Found theta:\n', theta_hinge)
    print('0/1 loss:', L01_loss(X_train_augmented, y_train, theta_hinge))

    if args.plot:
        plot_data_scatterplot(X_train, y_train,
                              [(best_theta, '$L_{01}$'),
                               (theta_square, '$L_2$'),
                               (theta_hinge, '$L_h$')])
