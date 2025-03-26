import numpy as np


def softmax_lastdim(x):
    """Compute softmax across last dimension of x.

    x is an arbitrary array with at least two dimensions. The returned array has
    the same shape as x, but its elements sum up to 1 across the last dimension.
    """
    # Subtract the max for numerical stability
    ex = np.exp(x - np.max(x, axis=-1, keepdims=True))
    # Divide by sums across last dimension
    return ex / np.sum(ex, axis=-1, keepdims=True)


def softmax_cols(x):
    """Compute softmax for each column of x.

    The result has the same shape as x, with each column replaced by the
    softmax of that column.
    """
    # Subtract the max for numerical stability
    ex = np.exp(x - np.max(x, axis=0, keepdims=True))
    # Divide by column-wise sums
    return ex / np.sum(ex, axis=0, keepdims=True)
