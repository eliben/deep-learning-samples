import numpy as np
from softmax import softmax_lastdim


def feed_forward_relu(x, W1, W2):
    """Feed-forward layer with ReLU activation.

    Args:
        x: Input tensor (B, N, D).
        Wh: Weights for the hidden layer (D, DH).
        Wo: Weights for the output layer (DH, D).

    Returns:
        Output tensor (B, N, D).
    """
    assert x.shape[-1] == W1.shape[0] == W2.shape[1]
    assert W1.shape[1] == W2.shape[0]

    x = np.maximum(0, x @ W1)  # (B, N, DH)
    return x @ W2  # (B, N, D)


def router(x, Wr):
    """Router (gate).

    Args:
        x: Input tensor (B, N, D).
        Wr: Router weight matrix (D, NEXP).

    Returns:
        Output tensor (B, N, NEXP).
    """
    assert x.shape[-1] == Wr.shape[0]
    return x @ Wr  # (B, N, NEXP)
