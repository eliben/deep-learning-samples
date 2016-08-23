# Helper code for downloading and unpickling MNIST data.
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

    If needed, downloads the data as a pickled .gz archive; This is my mirror of
    the archive originally taken from
    http://deeplearning.net/tutorial/gettingstarted.html

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


if __name__ == '__main__':
    train, valid, test = get_mnist_data()

    print('Train shapes:', train[0].shape, train[1].shape)
    print('Validation shapes:', validation[0].shape, validation[1].shape)
    print('Test shapes:', test[0].shape, test[1].shape)
