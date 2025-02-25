import numpy as np
import matplotlib.pyplot as plt


# Sinusoidal position encoding, as in the Transformer model paper.
# x is the inputs (N, D), each in a row.
# The output has the same shape.
def position_enc_sin_single(x):
    N, D = x.shape
    pos = np.arange(N).reshape(-1, 1)
    i = np.arange(0, D, 2)
    denom = 10000 ** (2 * i / D)
    enc = np.zeros((N, D))
    enc[:, 0::2] = np.sin(pos / denom)
    enc[:, 1::2] = np.cos(pos / denom)
    return x + enc


inp = np.zeros((96, 48))
out = position_enc_sin_single(inp).T

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3))
pos = ax.imshow(out, cmap="RdGy", extent=(1, out.shape[1] + 1, out.shape[0] + 1, 1))
fig.colorbar(pos, ax=ax)
ax.set_xlabel("position in sequence")
ax.set_ylabel("dimension")
plt.show()
