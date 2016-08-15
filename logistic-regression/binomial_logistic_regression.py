from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np


# TODO: shuffle data points before training?
def generate_data(n, num_neg_outliers=0):
    """Generates 2n data points: positives and negatives.

    Returns a pair of two (n,2) arrays: first is for positives, second for
    negatives. Out of n negatives, num_neg_outliers are outliers.

    Each data point is x coordinate, y coordinate
    """
    nneg_regular = n - num_neg_outliers
    negatives = (np.full((nneg_regular, 2), 1.0) +
                 np.random.normal(scale=0.7, size=(nneg_regular, 2)))
    positives = (np.full((n, 2), 3.0) +
                 np.random.normal(scale=0.9, size=(n, 2)))

    outliers = (np.full((num_neg_outliers, 2), 4.0) +
                np.random.normal(scale=1.2, size=(num_neg_outliers, 2)))

    return np.vstack((negatives, outliers)), positives


def plot_data_scatterplot(negatives, positives):
    """Plots data as a scatterplot.

    negatives: (n,2) array
    positives: (n,2) array

    Plots True data points as a green x, False as red o.
    """
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    ax.scatter(negatives[:, 0], negatives[:, 1], c='red', marker='o',
               linewidths=0)
    ax.scatter(positives[:, 0], positives[:, 1], c='darkgreen', marker='x')
    plt.show()


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


def predict(X, theta):
    yhat = X.dot(theta)
    return yhat >= 0


if __name__ == '__main__':
    neg, pos = generate_data(n=200, num_neg_outliers=10)
    print(neg)
    print(pos)
    plot_data_scatterplot(neg, pos)
    #print(generate_data(10))
    #pass
