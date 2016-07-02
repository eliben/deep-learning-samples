from __future__ import print_function

import time

import matplotlib.pyplot as plt
import numpy as np


def show_image(imgarr):
    """Given a numpy 2D array for a sample image, show and describe it.
    """
    print('Image type:', type(imgarr))
    print('Image shape:', imgarr.shape, '    dtype:', imgarr.dtype)
    if imgarr.ndim != 2:
        raise ValueError('Expected ndim=2')
    print('Image data:')
    for row in imgarr:
        for v in row:
            print('%6.2f ' % v, end='')
        print()
    plt.imshow(imgarr, cmap='gray')
    plt.show()


def shuffle_data_and_labels(dataset, labels):
    assert labels.ndim == 1
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s] ' % self.name, end='')
        print('Elapsed: %s' % (time.time() - self.tstart))
