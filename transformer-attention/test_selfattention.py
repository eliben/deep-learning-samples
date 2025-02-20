import numpy as np
import pytest
from selfattention import self_attention


def test_shapes():
    D = 8
    N = 12
    x = np.random.randn(D, N)
    Wk = np.random.randn(D, D)
    Wq = np.random.randn(D, D)
    Wv = np.random.randn(D, D)
    Bk = np.random.randn(D, 1)
    Bq = np.random.randn(D, 1)
    Bv = np.random.randn(D, 1)
    y = self_attention(x, Wk, Wq, Wv, Bk, Bq, Bv)
    assert y.shape == (D, N)
