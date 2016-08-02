# Example of solving multivariate linear regression in Python.
#
# Uses only Numpy, with Matplotlib for plotting.
#
# Eli Bendersky (http://eli.thegreenplace.net)
# This code is in the public domain
from __future__ import print_function
import csv
import matplotlib.pyplot as plt
import numpy as np

from timer import Timer


def read_CCPP_data(filename):
    """Read data from the given CCPP CSV file.

    Returns (data, header). data is a 2D Numpy array of type np.float32, with a
    sample per row. header is the names of the columns as read from the CSV
    file.
    """
    with open(filename, 'rb') as file:
        reader = csv.reader(file)
        header = reader.next()
        return np.array(list(reader), dtype=np.float32), header


def feature_normalize(X):
    """Normalize the feature matrix X.

    Given a feature matrix X, where each row is a vector of features, normalizes
    each feature. Returns (X_norm, mu, sigma) where mu and sigma are the mean
    and stddev of features (vectors).
    """
    num_features = X.shape[1]
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


def compute_cost(X, y, theta):
    """Compute the MSE cost of a prediction based on theta, over the whole X.

    X: (k, n) each row is an input with n features (including an all-ones
       column that should have been added beforehead).
    y: (k, 1) observed output per input.
    theta: (n, 1) regression parameters.

    Note: expects y and theta to be proper column vectors.
    """
    k = X.shape[0]
    # Vectorized computation of yhat per sample.
    yhat = np.dot(X, theta)
    diff = yhat - y
    # Vectorized computation using a dot product to compute sum of squares.
    cost = np.dot(diff.T, diff) / k
    # Cost is a 1x1 matrix, we need a scalar.
    return cost.flat[0]


def gradient_descent(X, y, nsteps, learning_rate=0.1):
    """Runs gradient descent optimization to fit a line y^ = theta.dot(x).

    X: (k, n) each row is an input with n features (including an all-ones column
       that should have been added beforehead).
    y: (k, 1) observed output per input.
    nsteps: how many steps to run the optimization for.
    learning_rate: learning rate of gradient descent.

    Yields 'nsteps + 1' pairs of (theta, cost) where theta is the fit parameter
    shaped (n, 1) for that step, and its cost vs the real y. The first pair has
    the initial theta and cost; the rest carry results after each of the
    iteration steps.
    """
    k, n = X.shape
    theta = np.zeros((n, 1))
    yield theta, compute_cost(X, y, theta)
    for step in range(nsteps):
        # yhat becomes a (k, 1) array of predictions, per sample.
        yhat = np.dot(X, theta)
        diff = yhat - y
        dtheta = np.zeros((n, 1))
        for j in range(n):
            # The sum over all samples is computed with a dot product between
            # (y^-y) and the jth feature across all of X.
            dtheta[j, 0] = learning_rate * np.dot(diff.T, X[:, j]) / k
        theta -= dtheta
        yield theta, compute_cost(X, y, theta)


def compute_normal_eqn(X, y):
    """Compute theta using the normal equation.

    X is the input matrix with a leftmost column of 1s. Returns theta as a
    column vector.
    """
    XTX = np.dot(X.T, X)
    # Using linalg.inv here, which will bomb for a singular matrix.
    # Alternatively, we could use linalg.pinv to compute a pseudo-inverse.
    XTX_inv = np.linalg.inv(XTX)
    xdot = np.dot(XTX_inv, X.T)
    return np.dot(xdot, y)


def split_dataset_to_train_test(dataset, train_proportion=0.8):
    """Splits the dataset to a train set and test set.

    The split is done over a random shuffle of the rows of the dataset. Assumes
    each row in the data has the expected outcome in the last column.

    train_proportion:
        The proportion of data to keep in the training set. The rest goes to the
        test set.

    Returns (X_train, y_train, X_test, y_test) where y_train/y_test are column
    vectors taken from the last column of the dataset.
    """
    shuffled_dataset = np.random.permutation(dataset)
    k_train = int(shuffled_dataset.shape[0] * train_proportion)

    X_train = shuffled_dataset[:k_train, :-1]
    y_train = shuffled_dataset[:k_train, -1].reshape(-1, 1)
    X_test = shuffled_dataset[k_train:, :-1]
    y_test = shuffled_dataset[k_train:, -1].reshape(-1, 1)
    return X_train, y_train, X_test, y_test


def compute_rsquared(X, y, theta):
    """Compute R^2 - the coefficeint of determination for theta.

    X: (k, n) input.
    y: (k, 1) observed output per input.
    theta: (n, 1) regression parameters.

    Returns the R2 - a scalar.
    """
    k = X.shape[0]
    yhat = np.dot(X, theta)
    diff = yhat - y
    SE_line = np.dot(diff.T, diff)
    SE_y = len(y) * y.var()
    return (1 - SE_line / SE_y).flat[0]


def plot_cost_vs_step(costs):
    """Given an array of costs, plots them vs. index.

    Uses logarithmic scale for y be cause the cost tends to be very large
    initially.
    """
    fig, ax = plt.subplots()
    ax.plot(range(len(costs)), costs)
    ax.set_yscale('log')
    plt.show()


def plot_correlation_heatmap(X, header):
    """Plot a heatmap of the correlation matrix for X.

    This requires the seaborn package to be installed.
    """
    import seaborn
    cm = np.corrcoef(X.T)
    hm = seaborn.heatmap(cm,
            cbar=True,
            annot=True,
            square=True,
            yticklabels=header,
            xticklabels=header)
    plt.show()


def sample_predictions_vs_truth(X, y, theta, nsamples=10):
    """Display a sample of predictions vs. true values."""
    print('Sample of predictions vs. true values')
    yhat = np.dot(X, theta)
    sample_indices = np.random.choice(X.shape[0], size=nsamples, replace=False)
    for index in sample_indices:
        print('  sample #{0}: yhat={1}, y={2}'.format(index,
                                                      yhat[index][0],
                                                      y[index][0]))


if __name__ == '__main__':
    # Follow through the code here to see how the functions are used. No
    # plotting is done by default. Uncomment relevant lines to produce plots.

    # For reproducibility
    np.random.seed(42)

    # This file was dowloaded from:
    # https://archive.ics.uci.edu/ml/machine-learning-databases/00294/ and then
    # unzipped.
    filename = 'CCPP-dataset/data.csv'
    with Timer('reading data'):
        X, header = read_CCPP_data(filename)

    # Plot a heatmap for the correlation matrix of X. This requires the seaborn
    # package. This heatmap is a useful visualization for finding features that
    # are most correlated with the result, and features that are possibly
    # collinear.
    #plot_correlation_heatmap(X, header)

    print('Read {0} data samples from {1}'.format(len(X), filename))
    X_train, y_train, X_test, y_test = split_dataset_to_train_test(X)
    print('Data shapes:')
    print('  X_train:', X_train.shape)
    print('  y_train:', y_train.shape)
    print('  X_test:', X_test.shape)
    print('  y_test:', y_test.shape)

    # Normalize X to bring all features into the same scale. Also, add a
    # all-ones column as the first column of X ("augmented X") to serve as the
    # bias term.
    ktrain = X_train.shape[0]
    X_train_normalized, mu, sigma = feature_normalize(X_train)
    X_train_augmented = np.hstack((np.ones((ktrain, 1)), X_train_normalized))

    # Run gradient descent.
    NSTEPS = 500
    with Timer('Running gradient descent ({0} steps)'.format(NSTEPS)):
        thetas_and_costs = list(gradient_descent(X_train_augmented,
                                                 y_train, NSTEPS))
    # Plot cost vs. step for the last 100 steps (the first steps have an
    # enourmous errors compared to the final steps).
    #plot_cost_vs_step([cost for _, cost in thetas_and_costs][:100])

    last_theta = thetas_and_costs[-1][0]
    print('Best theta found:', last_theta)

    print('Training set MSE:',
          compute_cost(X_train_augmented, y_train, last_theta))
    print('Training set R^2:',
          compute_rsquared(X_train_augmented, y_train, last_theta))

    # Normalize the test set using the mu/sigma computed from the training set,
    # and augment it with the bias column of 1s.
    ktest = X_test.shape[0]
    X_test_normalized = (X_test - mu) / sigma
    X_test_augmented = np.hstack((np.ones((ktest, 1)), X_test_normalized))
    print('Test set MSE:',
          compute_cost(X_test_augmented, y_test, last_theta))
    print('Test set R^2:',
          compute_rsquared(X_test_augmented, y_test, last_theta))

    # To assess how good the fit is, print out a random sample of predictions
    # for the test set compared to the real y values for these inputs.
    sample_predictions_vs_truth(X_test_augmented, y_test, last_theta)

    # Compute theta using the normal equation and report MST / R^2.
    theta_from_normal_eqn = compute_normal_eqn(X_train_augmented, y_train)
    print('Theta from normal equation:', theta_from_normal_eqn)
    print('Test set MSE / normal:',
          compute_cost(X_test_augmented, y_test, theta_from_normal_eqn))
    print('Test set R^2 / normal:',
          compute_rsquared(X_test_augmented, y_test, theta_from_normal_eqn))
