import numpy as np

def softmax_columns(x):
    # Subtract the max for numerical stability
    ex = np.exp(x - np.max(x, axis=0, keepdims=True))
    # Divide by column-wise sums
    return ex / np.sum(ex, axis=0, keepdims=True)

