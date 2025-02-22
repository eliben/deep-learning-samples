import numpy as np
from softmax import softmax_columns, softmax_rows


# self_attention the way it happens in the Transformer model.
# D = model dimension (length of embedding)
# N = input sequence length
#
# x is the inputs (N, D)
# Each of W* is a weight matrix of shape (D, D)
# Each of B* is a vector of shape (D, 1)
def self_attention(x, Wk, Wq, Wv, Bk, Bq, Bv):
    q = x @ Wq + Bq
    k = x @ Wk + Bk
    v = x @ Wv + Bv

    kq = q @ k.T / np.sqrt(k.shape[1])

    # att: (N, N) attention matrix. The rows become the weights that sum
    # to 1 for each output vector.
    att = softmax_rows(kq)
    return att @ v


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
