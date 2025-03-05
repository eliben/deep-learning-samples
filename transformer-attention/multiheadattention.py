import numpy as np
from softmax import softmax_lastdim


# x has shape (B, N, D)
# In what follows:
#   NH = number of heads
#   HS = head size
# Each W*s is a list of NH weight matrices of shape (D, HS).
# Wp is a weight matrix for the final linear projection, of shape (NH * HS, D)
# The result is (B, N, D)
# If do_mask is True, each attention head is masked from attending to future
# tokens.
def multihead_attention(x, Wks, Wqs, Wvs, Wp, do_mask=False):
    # Check shapes.
    NH = len(Wks)
    HS = Wks[0].shape[1]
    assert len(Wks) == len(Wqs) == len(Wvs)
    for W in Wqs + Wks + Wvs:
        assert W.shape[1] == HS
    assert Wp.shape[0] == NH * HS

    # List of head outputs
    head_outs = []

    if do_mask:
        # mask is a lower-triangular (N, N) matrix, with zeros above
        # the diagonal and ones on the diagonal and below.
        N = x.shape[1]
        mask = np.tril(np.ones((N, N)))

    for Wk, Wq, Wv in zip(Wks, Wqs, Wvs):
        # Calculate self attention for each head separately
        q = x @ Wq  # (B, N, HS)
        k = x @ Wk  # (B, N, HS)
        v = x @ Wv  # (B, N, HS)

        kq = q @ k.swapaxes(-2, -1) / np.sqrt(k.shape[-1])  # (B, N, N)

        if do_mask:
            # Set the masked positions to -inf, to ensure that a token isn't
            # affected by tokens that come after it in the softmax.
            kq = np.where(mask == 0, -np.inf, kq)

        att = softmax_lastdim(kq)  # (B, N, N)
        head_outs.append(att @ v)  # (B, N, HS)

    # Concatenate the head outputs and apply the final linear projection
    all_heads = np.concatenate(head_outs, axis=-1)  # (B, N, NH * HS)
    return all_heads @ Wp  # (B, N, D)
