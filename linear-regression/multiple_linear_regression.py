from __future__ import print_function
import csv
import matplotlib.pyplot as plt
import numpy as np

from timer import Timer

# TODO:
#
# - split to training + test set
# Try to train only high-correlation columns vs. all columns and report results.

def read_data(filename):
    """Read data from the given CSV file.

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
    k, n = X.shape
    theta = np.zeros((n, 1))
    yield theta, compute_cost(X, y, theta)

    for step in range(nsteps):
        yhat = np.dot(X, theta)
        diff = yhat - y
        dtheta = np.zeros((n, 1))
        for j in range(n):
            dtheta[j, 0] = learning_rate * np.dot(diff.T, X[:, j]) / k
        theta -= dtheta
        yield theta, compute_cost(X, y, theta)


def split_dataset_to_train_test(dataset, train_proportion=0.8):
    """Splits the dataset to a train set and test set.

    The split is done over a random shuffle of the rows of the dataset.

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


def plot_correlation_heatmap(X, header):
    """Plot a heatmap of the correlation matrix for X."""
    import seaborn
    cm = np.corrcoef(xnorm.T)
    hm = seaborn.heatmap(cm,
            cbar=True,
            annot=True,
            square=True,
            yticklabels=header,
            xticklabels=header)
    plt.show()


if __name__ == '__main__':
    # For reproducibility
    np.random.seed(42)

    filename = 'CCPP-dataset/data.csv'
    with Timer('reading data'):
        X, header = read_data(filename)
    print('Read {0} data samples from {1}'.format(len(X), filename))
    X_train, y_train, X_test, y_test = split_dataset_to_train_test(X)
    print('X_train:', X_train.shape)
    print('y_train:', y_train.shape)
    print('X_test:', X_test.shape)
    print('y_test:', y_test.shape)

    ktrain = X_train.shape[0]
    X_train_normalized, mu, sigma = feature_normalize(X_train)
    X_train_augmented = np.hstack((np.ones((ktrain, 1)), X_train_normalized))

    NSTEPS = 550
    for theta, cost in gradient_descent(X_train_augmented, y_train, NSTEPS):
        print(cost, "---->", list(theta.flat))
