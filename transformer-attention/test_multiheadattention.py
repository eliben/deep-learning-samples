import numpy as np
from multiheadattention import (
    multihead_attention,
    multihead_cross_attention,
    multihead_attention_vec,
)


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


_multihead_want = np.array(
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


def test_values_vec():
    D = 6
    NH = 3
    HS = 2
    B = 3
    N = 4
    x = np.linspace(0.1, 8.4, B * N * D).reshape(B, N, D)

    Wks = [np.linspace(i + 0.1, i + 0.8, D * HS).reshape(HS, D).T for i in range(NH)]
    Wqs = [np.linspace(i + 3.1, i + 3.8, D * HS).reshape(HS, D).T for i in range(NH)]
    Wvs = [np.linspace(i + 6.1, i + 6.8, D * HS).reshape(HS, D).T for i in range(NH)]
    Wp = np.linspace(9.1, 9.8, NH * HS * D).reshape(D, NH * HS).T
    print(Wks[0].shape)

    # concatenate all Ws, first K, then Q, then V
    Wk = np.concatenate(Wks, axis=1)
    Wq = np.concatenate(Wqs, axis=1)
    Wv = np.concatenate(Wvs, axis=1)
    W = np.concatenate([Wk, Wq, Wv], axis=1)

    y = multihead_attention_vec(x, W, NH, Wp)

    assert np.allclose(y, _multihead_want, rtol=1e-3)


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

    assert np.allclose(y, _multihead_want, rtol=1e-3)

    # Now test with masking
    y2 = multihead_attention(x, Wks, Wqs, Wvs, Wp, do_mask=True)

    want2 = np.array(
        [
            [
                [970.5113, 982.8517, 995.4508, 1008.5833, 1021.2378, 1033.4402],
                [2692.5103, 2727.4250, 2762.5979, 2798.3047, 2833.5334, 2868.3098],
                [4414.5088, 4471.9976, 4529.7451, 4588.0254, 4645.8281, 4703.1792],
                [6136.5073, 6216.5708, 6296.8916, 6377.7466, 6458.1230, 6538.0483],
            ],
            [
                [7858.5068, 7961.1440, 8064.0391, 8167.4678, 8270.4189, 8372.9180],
                [9580.5049, 9705.7168, 9831.1865, 9957.1885, 10082.7139, 10207.7871],
                [
                    11302.5049,
                    11450.2910,
                    11598.3350,
                    11746.9111,
                    11895.0098,
                    12042.6572,
                ],
                [
                    13024.5049,
                    13194.8643,
                    13365.4814,
                    13536.6338,
                    13707.3066,
                    13877.5273,
                ],
            ],
            [
                [
                    14746.5029,
                    14939.4375,
                    15132.6289,
                    15326.3545,
                    15519.6016,
                    15712.3965,
                ],
                [
                    16468.5039,
                    16684.0098,
                    16899.7754,
                    17116.0762,
                    17331.8965,
                    17547.2676,
                ],
                [
                    18190.5020,
                    18428.5820,
                    18666.9238,
                    18905.7969,
                    19144.1934,
                    19382.1348,
                ],
                [
                    19912.5020,
                    20173.1562,
                    20434.0703,
                    20695.5195,
                    20956.4883,
                    21217.0059,
                ],
            ],
        ]
    )

    assert np.allclose(y2, want2, rtol=1e-3)


def test_shapes_cross():
    # def multihead_cross_attention(xq, xv, Wks, Wqs, Wvs, Wp):
    D = 12
    Nq = 6
    Nv = 8
    HS = 3
    NH = 4
    B = 2
    xq = np.random.randn(B, Nq, D)
    xv = np.random.randn(B, Nv, D)
    Wks = [np.random.randn(D, HS) for _ in range(NH)]
    Wqs = [np.random.randn(D, HS) for _ in range(NH)]
    Wvs = [np.random.randn(D, HS) for _ in range(NH)]
    Wp = np.random.randn(NH * HS, D)

    y = multihead_cross_attention(xq, xv, Wks, Wqs, Wvs, Wp)
    assert y.shape == (B, Nq, D)
