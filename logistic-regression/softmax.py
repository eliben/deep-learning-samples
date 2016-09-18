# Experiments with softmax and its numerical stability
#
# Eli Bendersky (http://eli.thegreenplace.net)
# This code is in the public domain
import numpy as np


def softmax(x):
    """Compute the softmax of vector x."""
    exps = np.exp(x)
    return exps / np.sum(exps)


def stablesoftmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    x -= np.max(x)
    exps = np.exp(x)
    return exps / np.sum(exps)


if __name__ == '__main__':
    #pa = [2945.0, 2945.5]
    pa = [1000, 2000, 3000]
    print(softmax(pa))
    print(stablesoftmax(pa))
