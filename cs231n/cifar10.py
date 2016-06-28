# Access to the CIFAR-10 dataset.
#
# Some of the code is based on the cs231n data utils code
# [http://cs231n.github.io/]
# See https://www.cs.toronto.edu/~kriz/cifar.html for information on the
# data set and its format.
#
# * Xs (data) are arrays of 32x33x3 arrays of pixels. Logically, the values are
#   between 0-255, so they would fit into a uint8. However, since we want to
#   perform math on them without running into uint overflow and other problems,
#   we load them as float64.
# * ys (labels) are arrays of integers in the range 0-9
import cPickle as pickle
import numpy as np
import os


def _load_CIFAR_batch(filename):
    """Load a single batch of CIFAR from the given file."""
    with open(filename, 'rb') as f:
        datadict = pickle.load(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype('float64')
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(rootdir):
    """Load the whole CIFAR-10 data set.

    Given a path to the root directory containing CIFAR-10 samples in batches.
    Returns a 4-tuple: (Xtraining, Ytraining, Xtest, Ytest).
    """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(rootdir, 'data_batch_%d' % (b, ))
        X, Y = _load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    Xte, Yte = _load_CIFAR_batch(os.path.join(rootdir, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def show_CIFAR10_samples(X_train, y_train):
    """Show some sample images with classifications from the dataset."""
    import matplotlib.pyplot as plt
    classes = ['plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    samples_per_class = 7
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * len(classes) + y + 1
            plt.subplot(samples_per_class, len(classes), plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()


if __name__ == '__main__':
    dir = 'datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(dir)

    print 'Training data shape: ', X_train.shape, X_train.dtype
    print 'Training labels shape: ', y_train.shape, y_train.dtype
    print 'Test data shape: ', X_test.shape, X_test.dtype
    print 'Test labels shape: ', y_test.shape, y_test.dtype

    print 'Showing a few samples from the dataset.....'
    show_CIFAR10_samples(X_train, y_train)
