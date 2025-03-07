import numpy as np
from softmax import softmax_cols, softmax_lastdim


# self_attention the way it happens in the Transformer model. No bias.
# D = model dimension/depth (length of embedding)
# N = input sequence length
# HS = head size
#
# x is the input (N, D), each token in a row.
# Each of W* is a weight matrix of shape (D, HS)
# The result is (N, HS)
def self_attention(x, Wk, Wq, Wv):
    # Each of these is (N, D) @ (D, HS) = (N, HS)
    q = x @ Wq
    k = x @ Wk
    v = x @ Wv

    # kq: (N, N) matrix of dot products between each pair of q and k vectors.
    # The division by sqrt(D) is the scaling.
    kq = q @ k.T / np.sqrt(k.shape[1])

    # att: (N, N) attention matrix. The rows become the weights that sum
    # to 1 for each output vector.
    att = softmax_lastdim(kq)
    return att @ v  # (N, HS)


# self_attention with inputs that have a batch dimension.
# x has shape (B, N, D)
# Each of W* has shape (D, HS)
def self_attention_batched(x, Wk, Wq, Wv):
    q = x @ Wq  # (B, N, HS)
    k = x @ Wk  # (B, N, HS)
    v = x @ Wv  # (B, N, HS)

    kq = q @ k.swapaxes(-2, -1) / np.sqrt(k.shape[-1])  # (B, N, N)

    att = softmax_lastdim(kq)  # (B, N, N)
    return att @ v  # (B, N, HS)


# D = model dimension (length of embedding)
# N = input sequence length
#
# x is the inputs (D, N)
# Each of W* is a weight matrix of shape (D, D)
# Each of B* is a vector of shape (D, 1)
def self_attention_cols(x, Wk, Wq, Wv, Bk, Bq, Bv):
    q = Wq @ x + Bq
    k = Wk @ x + Bk
    v = Wv @ x + Bv

    kq = (k.T @ q) / np.sqrt(k.shape[0])

    # att: (N, N) attention matrix. The columns become the weights that sum
    # to 1 for each output vector.
    att = softmax_cols(kq)
    return v @ att
