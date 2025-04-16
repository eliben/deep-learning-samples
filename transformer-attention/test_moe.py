import numpy as np
from moe import topk_lastdim


def test_topk_lastdim():
    np.random.seed(12)
    B = 6
    N = 4
    D = 8

    x = np.random.randn(B, N, D)

    k = 3
    y, idx = topk_lastdim(x, k)
    assert y.shape == (B, N, k)
    assert idx.shape == (B, N, k)

    for i in range(B):
        for j in range(N):
            # Get the top k values and their indices
            top_k_values = np.partition(x[i, j], -k)[-k:]
            top_k_indices = np.argpartition(x[i, j], -k)[-k:]
            assert np.allclose(y[i, j], top_k_values)
            assert np.all(idx[i, j] == top_k_indices)
