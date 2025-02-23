import numpy as np
from softmax import softmax_rows


# x has shape (B, N, D)
# Each W*s is a list of NH weight matrices of shape (D, HS) where HS is the head
# size. NH and HS must be the same across all lists.
# Wp is a weight matrix for the final linear projection, of shape (NH * HS, D)
def multihead_attention(x, Wks, Wqs, Wvs, Wp):
    # Check shapes.
    assert len(Wks) == len(Wqs) == len(Wvs)
    NH = len(Wks)
    HS = Wks[0].shape[1]
    for W in Wqs + Wks + Wvs:
        assert W.shape[1] == HS
    assert Wp.shape[0] == NH * HS

    # List of head outputs
    head_outs = []

    for Wk, Wq, Wv in zip(Wks, Wqs, Wvs):
        # Calculate self attention for each head separately
        q = x @ Wq  # (B, N, HS)
        k = x @ Wk  # (B, N, HS)
        v = x @ Wv  # (B, N, HS)

        kq = q @ k.swapaxes(-2, -1) / np.sqrt(k.shape[-1])  # (B, N, N)
        att = softmax_rows(kq)  # (B, N, N)
        head_outs.append(att @ v)  # (B, N, HS)

    # Concatenate the head outputs and apply the final linear projection
    all_heads = np.concatenate(head_outs, axis=-1)  # (B, N, NH * HS)
    return all_heads @ Wp  # (B, N, D)
