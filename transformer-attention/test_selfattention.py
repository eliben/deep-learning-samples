import numpy as np
import pytest
from selfattention import *


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


def test_batched():
    B = 3
    N = 4
    D = 2

    # This is for testing shape
    x = np.random.randn(B, N, D)
    Wk = np.random.randn(D, D)
    Wq = np.random.randn(D, D)
    Wv = np.random.randn(D, D)
    y = self_attention_batched(x, Wk, Wq, Wv)
    assert y.shape == (B, N, D)

    # Testing values
    x = np.linspace(0.1, 5.4, B * N * D).reshape(B, N, D)
    Wk = np.linspace(0.1, 0.8, D * D).reshape((D, D)).T
    Wq = np.linspace(1.1, 1.8, D * D).reshape((D, D)).T
    Wv = np.linspace(2.1, 2.8, D * D).reshape((D, D)).T
    y = self_attention_batched(x, Wk, Wq, Wv)

    want = np.array(
        [
            [[5.0515, 6.1093], [6.3565, 7.6890], [6.8308, 8.2633], [6.9991, 8.4669]],
            [
                [15.2371, 18.4392],
                [15.2638, 18.4716],
                [15.2750, 18.4852],
                [15.2798, 18.4909],
            ],
            [
                [23.4546, 28.3867],
                [23.4554, 28.3878],
                [23.4558, 28.3882],
                [23.4560, 28.3884],
            ],
        ]
    )
    assert np.allclose(y, want, atol=1e-3)
