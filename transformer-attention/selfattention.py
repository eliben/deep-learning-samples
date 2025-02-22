import numpy as np
from softmax import softmax_columns, softmax_rows


# self_attention the way it happens in the Transformer model. No bias.
# D = model dimension/depth (length of embedding)
# N = input sequence length
#
# x is the inputs (N, D), each in a row.
# Each of W* is a weight matrix of shape (D, D)
def self_attention(x, Wk, Wq, Wv):
    q = x @ Wq
    k = x @ Wk
    v = x @ Wv

    kq = q @ k.T / np.sqrt(k.shape[1])

    # att: (N, N) attention matrix. The rows become the weights that sum
    # to 1 for each output vector.
    att = softmax_rows(kq)
    return att @ v


# self_attention with inputs that have a batch dimension.
# x has shape (B, N, D)
# Each of W* has shape (D, D)
def self_attention_batched(x, Wk, Wq, Wv):
    q = x @ Wq  # (B, N, D)
    k = x @ Wk  # (B, N, D)
    v = x @ Wv  # (B, N, D)

    kq = q @ k.swapaxes(-2, -1) / np.sqrt(k.shape[-1])  # (B, N, N)

    att = softmax_rows(kq)  # (B, N, N)
    return att @ v  # (B, N, D)


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
    att = softmax_columns(kq)
    return v @ att
