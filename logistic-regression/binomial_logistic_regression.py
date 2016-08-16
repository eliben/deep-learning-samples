from __future__ import print_function
import argparse
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


def plot_data_scatterplot(negatives, positives, theta=None):
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

    if theta is not None:
        xs = np.linspace(-2, 6, 200)
        ys = np.linspace(-2, 6, 200)
        xsgrid, ysgrid = np.meshgrid(xs, ys)
        plane = np.zeros_like(xsgrid)
        for i in range(xsgrid.shape[0]):
            for j in range(xsgrid.shape[1]):
                plane[i, j] = np.array([1, xsgrid[i, j], ysgrid[i, j]]).dot(
                    theta)
        ax.contour(xsgrid, ysgrid, plane, levels=[0])

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
    """Make classification predictions for the data in X using theta.

    X: (k, n) k rows of data items, each having n features; augmented.
    theta: (n, 1) regression parameters.

    Returns yhat (k, 1) - either +1 or -1 classification for each item.
    """
    yhat = X.dot(theta)
    return np.sign(yhat)


def L01_loss(X, y, theta):
    """Compute the L0/1 loss for the data X using theta.
    
    X: (k, n) k rows of data items, each having n features; augmented.
    y: (k, 1) correct classifications (+1 or -1) for each item.
    theta: (n, 1) regression parameters.

    Returns the total L0/1 loss over the whole data set. The total L0/1 loss
    is the number of mispredicted items (where y doesn't match yhat).
    """
    results = predict(X, theta)
    return np.count_nonzero(results != y)


def search_best_L01_loss(X, y, theta_start=None, npoints_per_t=20):
    if theta_start is None:
        theta_start = np.array([[1], [1], [1]])

    k = X.shape[0]
    best_loss = k
    best_theta = theta_start

    assert theta_start.shape == (3, 1)
    t0_range = np.linspace(theta_start[0, 0] + 5, theta_start[0, 0] - 5,
                           npoints_per_t)
    t1_range = np.linspace(theta_start[1, 0] + 5, theta_start[1, 0] - 5,
                           npoints_per_t)
    t2_range = np.linspace(theta_start[2, 0] + 5, theta_start[2, 0] - 5,
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


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--plot', action='store_true', required=False)
    args = argparser.parse_args()

    # For reproducibility
    np.random.seed(42)

    neg, pos = generate_data(n=200, num_neg_outliers=10)
    theta = np.array([-5, 2, 1]).reshape(-1, 1)

    if args.plot:
        plot_data_scatterplot(neg, pos, theta)

    # Attach labels (1.0 for positive, -1.0 for negative) to the data, so that
    # we can shuffle it together with the labels.
    pos = np.hstack((pos, np.full((pos.shape[0], 1), 1.0)))
    neg = np.hstack((neg, np.full((neg.shape[0], 1), -1.0)))
    full_dataset = np.random.permutation(np.vstack((pos, neg)))
    X_train = full_dataset[:, 0:2]
    y_train = full_dataset[:, 2].reshape(-1, 1)

    NORMALIZE = False

    if NORMALIZE:
        X_train_normalized, mu, sigma = feature_normalize(X_train)
    else:
        X_train_normalized, mu, sigma = X_train, 0, 1

    X_train_augmented = np.hstack((np.ones((X_train.shape[0], 1)),
                                           X_train_normalized))

    print('total L01 loss:', L01_loss(X_train_augmented, y_train, theta))

    best_theta, best_loss = search_best_L01_loss(X_train_augmented, y_train,
                                                 theta)

    print(best_theta)
    print(best_loss)
    #results = predict(X_train_augmented, theta)
    #print(results.shape)
    #print(X_train_augmented[:10])
    #for i in range(10):
        #print(y_train[i, 0], ' <> ', results[i, 0])
    #print(results[:40])
    #print(np.count_nonzero(results == y_train))
    #print(neg)
    #print(pos)

    #print(generate_data(10))
    #pass
