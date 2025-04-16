from dataclasses import dataclass
from typing import List

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


def topk_lastdim(x, k):
    """Get the top k elements and their indices.

    x is an arbitrary array with at least two dimensions. The returned
    array has the same shape as x, but its elements are the top k elements
    across the last dimension. The indices of the top k elements are also
    returned.
    """
    idx = np.argpartition(x, -k, axis=-1)[..., -k:]
    return np.take_along_axis(x, idx, axis=-1), idx


@dataclass
class FFParams:
    Wh: np.ndarray
    Wo: np.ndarray


@dataclass
class MoEParams:
    # Embedding dimension of each token
    D: int

    # Hidden dimension in FF layers
    DH: int

    # Total number of experts
    NEXP: int

    # K in the top-k selection of top experts per token
    TOPK: int

    ff_weights: List[FFParams]
    router_weights: np.ndarray


def moe(x, params):
    # Run input through router to get scores for each expert for each token.
    expert_scores = router(x, params.router_weights)  # (B, N, NEXP)

    # Select the top-k expert scores and their indices for each token.
    top_scores, top_experts = topk_lastdim(expert_scores, params.TOPK)  # (B, N, TOPK)

    # Apply softmax to the top scores to get weights that sum to 1.
    weights = softmax_lastdim(top_scores)  # (B, N, TOPK)

    out = np.zeros_like(x)  # Initialize output tensor (B, N, D)
    for b in range(x.shape[0]):
        for n in range(x.shape[1]):
            # Unvectorized implementation: for each token in the batch and
            # sequence, select the top-k experts and apply them with the
            # calculated weights.
            for expert_idx, weight in zip(top_experts[b, n], weights[b, n]):
                expert = params.ff_weights[expert_idx]
                out[b, n] += weight * feed_forward_relu(x[b, n], expert.Wh, expert.Wo)

    return out


if __name__ == "__main__":
    # Example usage
    B = 4
    N = 6
    D = 8
    DH = 16
    NEXP = 4
    TOPK = 2

    x = np.random.randn(B, N, D).astype(np.float32)  # Input tensor

    # Initialize parameters
    ff_weights = [
        FFParams(np.random.randn(D, DH), np.random.randn(DH, D)) for _ in range(NEXP)
    ]
    router_weights = np.random.randn(D, NEXP)

    params = MoEParams(
        D=D,
        DH=DH,
        NEXP=NEXP,
        TOPK=TOPK,
        ff_weights=ff_weights,
        router_weights=router_weights,
    )

    y = moe(x, params)
    print("Output shape:", y.shape)  # Should be (B, N, D)
    print("Output:", y)
