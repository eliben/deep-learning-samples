import numpy as np
import pytest
from multiheadattention import *


def test_shapes():
    # 4 heads (NH), each with depth 3 (H). Total H*NH=D=12
    D = 12
    N = 8
    HS = 3
    NH = 4
    B = 2
    x = np.random.randn(B, N, D)
    Wks = [np.random.randn(D, HS) for _ in range(NH)]
    Wqs = [np.random.randn(D, HS) for _ in range(NH)]
    Wvs = [np.random.randn(D, HS) for _ in range(NH)]
    Wp = np.random.randn(NH * HS, D)

    y = multihead_attention(x, Wks, Wqs, Wvs, Wp)
    assert y.shape == (B, N, D)


def test_values():
    D = 6
    NH = 3
    HS = 2
    B = 3
    N = 4
    x = np.linspace(0.1, 8.4, B * N * D).reshape(B, N, D)

    # As before, we have to reshape these into the transposed form when filling
    # them in, then transpose. This is because PyTorch transposes the weight
    # matrix directly assigned to a layer.
    Wks = [np.linspace(i + 0.1, i + 0.8, D * HS).reshape(HS, D).T for i in range(NH)]
    Wqs = [np.linspace(i + 3.1, i + 3.8, D * HS).reshape(HS, D).T for i in range(NH)]
    Wvs = [np.linspace(i + 6.1, i + 6.8, D * HS).reshape(HS, D).T for i in range(NH)]
    Wp = np.linspace(9.1, 9.8, NH * HS * D).reshape(D, NH * HS).T

    y = multihead_attention(x, Wks, Wqs, Wvs, Wp)

    want = np.array(
        [
            [
                [6136.3379, 6216.7163, 6297.3208, 6377.4688, 6458.0137, 6538.4683],
                [6136.3379, 6216.7163, 6297.3208, 6377.4688, 6458.0137, 6538.4683],
                [6136.3379, 6216.7163, 6297.3208, 6377.4688, 6458.0137, 6538.4683],
                [6136.3379, 6216.7163, 6297.3208, 6377.4688, 6458.0137, 6538.4683],
            ],
            [
                [
                    13024.3350,
                    13195.0098,
                    13365.9102,
                    13536.3564,
                    13707.1973,
                    13877.9473,
                ],
                [
                    13024.3350,
                    13195.0098,
                    13365.9102,
                    13536.3564,
                    13707.1973,
                    13877.9473,
                ],
                [
                    13024.3350,
                    13195.0098,
                    13365.9102,
                    13536.3564,
                    13707.1973,
                    13877.9473,
                ],
                [
                    13024.3350,
                    13195.0098,
                    13365.9102,
                    13536.3564,
                    13707.1973,
                    13877.9473,
                ],
            ],
            [
                [
                    19912.3320,
                    20173.3027,
                    20434.5000,
                    20695.2402,
                    20956.3789,
                    21217.4258,
                ],
                [
                    19912.3320,
                    20173.3027,
                    20434.5000,
                    20695.2402,
                    20956.3789,
                    21217.4258,
                ],
                [
                    19912.3320,
                    20173.3027,
                    20434.5000,
                    20695.2402,
                    20956.3789,
                    21217.4258,
                ],
                [
                    19912.3320,
                    20173.3027,
                    20434.5000,
                    20695.2402,
                    20956.3789,
                    21217.4258,
                ],
            ],
        ]
    )

    assert np.allclose(y, want, rtol=1e-3)
