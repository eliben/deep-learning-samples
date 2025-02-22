import numpy as np


def softmax_columns(x):
    """Compute softmax for each column of x.

    The result has the same shape as x, with each column replaced by the
    softmax of that column.
    """
    # Subtract the max for numerical stability
    ex = np.exp(x - np.max(x, axis=0, keepdims=True))
    # Divide by column-wise sums
    return ex / np.sum(ex, axis=0, keepdims=True)


def softmax_rows(x):
    """Compute softmax for each row of x.

    The input x has shape (N, M), where N is the number of rows and M is the
    number of columns. Or it can have more leading dimensions, but the softmax
    is applied to the last dimension.

    The result has the same shape as x, with the values replaced by the softmax
    calculate across the last dimension.
    """
    # Subtract the max for numerical stability
    ex = np.exp(x - np.max(x, axis=-1, keepdims=True))
    # Divide by row-wise sums
    return ex / np.sum(ex, axis=-1, keepdims=True)
