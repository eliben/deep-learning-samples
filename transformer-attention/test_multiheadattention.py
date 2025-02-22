import numpy as np
import pytest
from multiheadattention import *


def test_shapes():
    # 4 heads (NH), each with depth 3 (H). Total H*NH=D=12
    D = 12
    N = 8
    H = 3
    NH = 4
    B = 2
    x = np.random.randn(B, N, D)
    Wks = [np.random.randn(D, H) for _ in range(NH)]
    Wqs = [np.random.randn(D, H) for _ in range(NH)]
    Wvs = [np.random.randn(D, H) for _ in range(NH)]
    Wp = np.random.randn(NH * H, D)
    y = multihead_attention(x, Wks, Wqs, Wvs, Wp)
    assert y.shape == (B, N, D)
