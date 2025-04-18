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

    x = x @ W1  # hidden layer (B, N, DH)
    x = np.maximum(0, x)  # ReLU activation (B, N, DH)
    x = x @ W2  # output layer (B, N, D)
    return x


def topk_lastdim(x, k):
    """Get the top k elements and their indices.

    x is an arbitrary array with at least two dimensions. The returned
    array has the same shape as x, but its elements are the top k elements
    across the last dimension. The indices of the top k elements are also
    returned.
    """
    idx = np.argpartition(x, -k, axis=-1)[..., -k:]
    return np.take_along_axis(x, idx, axis=-1), idx


# Parameters for a feed-forward layer with a fixed activation function.
@dataclass
class FFParams:
    Wh: np.ndarray
    Wo: np.ndarray


# Parameters for a Mixture of Experts (MoE) layer.
@dataclass
class MoEParams:
    # Embedding dimension of each token (a.k.a. model dimension, Dmodel)
    D: int

    # Hidden dimension in FF layers
    DH: int

    # Total number of experts
    NEXP: int

    # K in the top-k selection of top experts per token
    TOPK: int

    # List of experts: each expert is a forward layer with FFParams.
    ff_weights: List[FFParams]

    # Router weights: a linear layer (D, NEXP) that maps input to expert scores.
    router_weights: np.ndarray


def moe(x: np.ndarray, params: MoEParams):
    """Mixture of Experts (MoE) layer.

    Args:
        x: Input tensor (B, N, D).
        params: MoEParams.

    Returns:
        Output tensor (B, N, D).
    """
    assert x.shape[-1] == params.D
    assert params.router_weights.shape == (params.D, params.NEXP)
    assert len(params.ff_weights) == params.NEXP
    for expert in params.ff_weights:
        assert expert.Wh.shape == (params.D, params.DH)
        assert expert.Wo.shape == (params.DH, params.D)

    # Run input through router to get expert scores for each token.
    expert_scores = x @ params.router_weights  # (B, N, NEXP)

    # Select the top-k expert scores and their indices for each token.
    top_scores, top_experts = topk_lastdim(expert_scores, params.TOPK)  # (B, N, TOPK)

    # Apply softmax to the top scores to get weights that sum to 1.
    weights = softmax_lastdim(top_scores)  # (B, N, TOPK)

    out = np.zeros_like(x)
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
    B = 4
    N = 6
    D = 8
    DH = 16
    NEXP = 4
    TOPK = 2

    params = MoEParams(
        D=D,
        DH=DH,
        NEXP=NEXP,
        TOPK=TOPK,
        ff_weights=[
            FFParams(np.random.randn(D, DH), np.random.randn(DH, D))
            for _ in range(NEXP)
        ],
        router_weights=np.random.randn(D, NEXP),
    )

    x = np.linspace(0.1, 8.4, B * N * D).reshape(B, N, D)

    y = moe(x, params)
    print("Output shape:", y.shape)  # Should be (B, N, D)
    print("Output:", y)
