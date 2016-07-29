from __future__ import print_function
import csv
import matplotlib.pyplot as plt
import numpy as np

from timer import Timer

# TODO:
#
# - In test, compare with sklearn's fit for the normalization?

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
    """Runs gradient descent optimization to fit a line y^ = theta.dot(x).

    X: (k, n) each row is an input with n features (including an all-ones column
       that should have been added beforehead).
    y: (k, 1) observed output per input.
    nsteps: how many steps to run the optimization for.
    learning_rate: learning rate of gradient descent.

    Yields 'nsteps + 1' pairs of (theta, cost) where theta is the fit parameter
    shaped (n, 1) for that step, and its cost vs the real y.
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


def compute_rsquared(X, y, theta):
    """Compute R^2 - the coefficeint of determination for theta.

    X: (k, n) input.
    y: (k, 1) observed output per input.
    theta: (n, 1) regression parameters.
    """
    k = X.shape[0]
    yhat = np.dot(X, theta)
    diff = yhat - y
    SE_line = np.dot(diff.T, diff)
    SE_y = len(y) * y.var()
    return (1 - SE_line / SE_y).flat[0]


def plot_cost_vs_step(costs):
    fig, ax = plt.subplots()
    ax.plot(range(len(costs)), costs)
    ax.set_yscale('log')
    plt.show()


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


def sample_predictions_vs_truth(X, y, theta, nsamples=10):
    """Display a sample of predictions vs. true values."""
    print('Sample of predictions vs. true values')
    yhat = np.dot(X, theta)
    sample_indices = np.random.choice(X.shape[0], size=nsamples, replace=False)
    for index in sample_indices:
        print('{0}: yhat={1}, y={2}'.format(index, yhat[index][0], y[index][0]))


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

    NSTEPS = 385
    with Timer('Running gradient descent ({0} steps)'.format(NSTEPS)):
        thetas_and_costs = list(gradient_descent(X_train_augmented,
                                                 y_train, NSTEPS))
    #plot_cost_vs_step([cost for _, cost in thetas_and_costs][:100])

    last_theta = thetas_and_costs[-1][0]
    print(last_theta)

    print('Training set MSE:',
          compute_cost(X_train_augmented, y_train, last_theta))
    print('Training set R^2:',
          compute_rsquared(X_train_augmented, y_train, last_theta))

    ktest = X_test.shape[0]
    X_test_normalized = (X_test - mu) / sigma
    X_test_augmented = np.hstack((np.ones((ktest, 1)), X_test_normalized))
    print('Test set MSE:',
          compute_cost(X_test_augmented, y_test, last_theta))
    print('Test set R^2:',
          compute_rsquared(X_test_augmented, y_test, last_theta))

    sample_predictions_vs_truth(X_test_augmented, y_test, last_theta)
