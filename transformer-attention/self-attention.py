import numpy as np


# D = model dimension (length of embedding)
# N = input sequence length
#
# x is the inputs (D, N)
# Each of W* is a weight matrix of shape (D, D)
# Each of B* is a vector of shape (D, 1)
def self_attention(x, Wk, Wq, Wv, Bk, Bq, Bv):
    q = Wq @ x + Bq
    k = Wk @ x + Bk
    v = Wv @ x + Bv

    kq = (k.T @ q) / np.sqrt(k.shape[0])
    att = softmax_columns(kq)
    return att @ v
