import numpy as np
import matplotlib.pyplot as plt


# Sinusoidal position encoding, as in the Transformer model paper.
# x is the inputs (N, D), each in a row.
# The output has the same shape.
def position_enc_sin_single(x):
    # PE(pos, 2i) = sin(pos / 10000^(2i/D))
    # PE(pos, 2i+1) = cos(pos / 10000^(2i/D))
    N, D = x.shape

    # pos is a column vector of shape (N, 1) with the position in the sequence.
    # i is a row vector of shape (1, D) with the dimension index. denom will
    # be the same shape as i.
    # pos / denom is a broadcast division of shape (N, D).
    pos = np.arange(N).reshape(-1, 1)
    i = np.arange(0, D, 2)
    denom = 10000 ** (2 * i / D)
    penc = np.zeros((N, D))
    penc[:, 0::2] = np.sin(pos / denom)
    penc[:, 1::2] = np.cos(pos / denom)
    return x + penc


# Batched version; x shape is (B, N, D).
def position_enc_sin(x):
    B, N, D = x.shape

    # penc is exactly the same as in position_enc_sin_single, and the final
    # addition will properly broadcast it over the B axis.
    pos = np.arange(N).reshape(-1, 1)
    i = np.arange(0, D, 2)
    denom = 10000 ** (2 * i / D)
    penc = np.zeros((N, D))
    penc[:, 0::2] = np.sin(pos / denom)
    penc[:, 1::2] = np.cos(pos / denom)
    return x + penc


inp = np.zeros((3, 120, 48))
out = position_enc_sin(inp)[0, :, :].T

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3))
pos = ax.imshow(out, extent=(1, out.shape[1] + 1, out.shape[0] + 1, 1))
fig.colorbar(pos, ax=ax)
ax.set_xlabel("position in sequence")
ax.set_ylabel("dimension")
plt.show()
