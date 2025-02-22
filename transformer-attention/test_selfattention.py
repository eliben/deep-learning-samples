import numpy as np
import pytest
from selfattention import self_attention, self_attention_cols


def test_shapes_cols():
    D = 8
    N = 12
    x = np.random.randn(D, N)
    Wk = np.random.randn(D, D)
    Wq = np.random.randn(D, D)
    Wv = np.random.randn(D, D)
    Bk = np.random.randn(D, 1)
    Bq = np.random.randn(D, 1)
    Bv = np.random.randn(D, 1)
    y = self_attention_cols(x, Wk, Wq, Wv, Bk, Bq, Bv)
    assert y.shape == (D, N)


def test_shapes_rows():
    D = 8
    N = 12
    x = np.random.randn(N, D)
    Wk = np.random.randn(D, D)
    Wq = np.random.randn(D, D)
    Wv = np.random.randn(D, D)
    y = self_attention(x, Wk, Wq, Wv)
    assert y.shape == (N, D)


def test_rows_values():
    # These expected values are taken from a Pytorch implementation of
    # self-attention.
    N = 6
    D = 4
    x = np.linspace(0.1, 2.4, N * D).reshape(N, D)
    print(x)

    # These have to be transposed to compare to torch because torch uses the
    # transposed weight matrix in a linear layer.
    Wk = np.linspace(0.1, 0.4, D * D).reshape((D, D)).T
    Wq = np.linspace(1.1, 1.4, D * D).reshape((D, D)).T
    Wv = np.linspace(2.1, 2.4, D * D).reshape((D, D)).T
    y = self_attention(x, Wk, Wq, Wv)

    want = np.array(
        [
            [17.3399, 17.9907, 18.6416, 19.2925],
            [18.9277, 19.6382, 20.3487, 21.0592],
            [19.1339, 19.8521, 20.5704, 21.2887],
            [19.1712, 19.8908, 20.6105, 21.3302],
            [19.1783, 19.8982, 20.6182, 21.3381],
            [19.1797, 19.8997, 20.6196, 21.3396],
        ]
    )
    assert np.allclose(y, want, atol=1e-3)
