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
    Bk = np.random.randn(1, D)
    Bq = np.random.randn(1, D)
    Bv = np.random.randn(1, D)
    y = self_attention(x, Wk, Wq, Wv, Bk, Bq, Bv)
    assert y.shape == (N, D)


def test_rows_values():
    N = 6
    D = 4
    x = np.linspace(0.1, 2.4, N * D).reshape(N, D)
    print(x)

    # These have to be transposed to compare to torch because torch uses the
    # transposed weight matrix in a linear layer.
    Wk = np.linspace(0.1, 0.4, D * D).reshape((D, D)).T
    Wq = np.linspace(1.1, 1.4, D * D).reshape((D, D)).T
    Wv = np.linspace(2.1, 2.4, D * D).reshape((D, D)).T
    print(Wk)
    print(Wq)
    print(Wv)
    Bk = np.zeros((1, D))
    Bq = np.zeros((1, D))
    Bv = np.zeros((1, D))
    y = self_attention(x, Wk, Wq, Wv, Bk, Bq, Bv)
    print(y)
