# Helper code for downloading, unpickling and displaying MNIST data.
#
# Eli Bendersky (http://eli.thegreenplace.net)
# This code is in the public domain
from __future__ import print_function
import cPickle as pickle
import gzip
import os
from shutil import copyfileobj
from urllib2 import urlopen
from urlparse import urljoin

import matplotlib.pyplot as plt
import numpy as np


def maybe_download(base_url, filename, expected_size, force=False):
    """Download a file if not present, and make sure it's the right size."""
    if force or not os.path.exists(filename):
        print('Attempting to download:', filename)
        in_stream = urlopen(urljoin(base_url, filename))
        with open(filename, 'wb') as out_file:
            copyfileobj(in_stream, out_file)
        print('Download Complete!')
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_size:
        print('Found and verified', filename)
        return True
    else:
        print('Unable to verify size: {0} vs. expected {1}'.format(
            statinfo.st_size, expected_size))
        return False


def load_pickle_from_gz(filename):
    """Load a pickle from a gzip archive."""
    with gzip.open(filename, 'rb') as f:
        return pickle.loads(f.read())


def get_mnist_data():
    """Get data sets for MNIST.

    If needed, downloads the data as a pickled .gz archive; Taken from my mirror
    of the archive at http://deeplearning.net/tutorial/gettingstarted.html.

    The pickle contains 3 sets in a tuple: training, validation and test data
    sets. Each data set is a pair of numpy arrays: data (N x 784) and numeric
    labels (N,) where N is the set size.
    """
    baseurl = 'http://thegreenplace.net/files/'
    filename = 'mnist.pkl.gz'
    if maybe_download(baseurl, filename, expected_size=16168813):
        return load_pickle_from_gz(filename)
    else:
        return None


def display_mnist_image(x, y=None):
    """Displays a single mnist image with a label.

    x: (784,) image vector, as stored in the mnist pickle.
    y: optional numeric label
    """
    xmat = x.reshape(28, 28)
    plt.imshow(xmat, cmap='gray')
    if y is not None:
        plt.title('label={0}'.format(y))
    plt.show()


def display_multiple_images(xs):
    """Displays multiple images side-by-side in subplots."""
    fig = plt.figure()
    fig.set_tight_layout(True)

    for i, x in enumerate(xs):
        ax = fig.add_subplot(1, len(xs), i + 1)
        ax.imshow(x.reshape(28, 28), cmap='gray')
    plt.show()


def convert_y_to_binary(y, correct_digit):
    """Converts a vector y taken from MNIST data to binary "is it this digit".

    y: array of digits.
    correct_digit: the digit we expect to be "correct"

    Returns array of +1 or -1; +1 where the original y had the "correct" digit,
    and -1 otherwise. The returned array is always a column vector.
    """
    return np.where(y == correct_digit,
                    np.ones_like(y),
                    -1 * np.ones_like(y)).reshape(y.size, 1)


if __name__ == '__main__':
    train, valid, test = get_mnist_data()

    print('Train shapes:', train[0].shape, train[1].shape)
    print('Valid shapes:', valid[0].shape, valid[1].shape)
    print('Test shapes:', test[0].shape, test[1].shape)

    #display_mnist_image(train[0][20], train[1][20])

    display_multiple_images((train[0][9974],
                             train[0][9734],
                             train[0][9161],
                             train[0][8788]))
