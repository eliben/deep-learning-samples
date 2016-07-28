from __future__ import print_function
import csv
import matplotlib.pyplot as plt
import numpy as np

from timer import Timer

# TODO:
#
# - feature normalization
# - split to training + test set

def read_data(filename):
    """Read data from the given CSV file.

    Returns a 2D Numpy array of type np.float32, with a sample per row.
    """
    with open(filename, 'rb') as file:
        reader = csv.reader(file)
        reader.next() # skip header
        return np.array(list(reader), dtype=np.float32)


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


if __name__ == '__main__':
    filename = 'CCPP-dataset/data.csv'
    with Timer('reading data'):
        data = read_data(filename)
    print('Read {0} data samples from {1}'.format(len(data), filename))
    print(data[0:4])
    xnorm, mu, sigma = feature_normalize(data)
    print(mu, sigma)
    print(xnorm[0:4])
