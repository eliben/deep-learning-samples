import torch
import torch.nn as nn

M = 6
N = 4
K = 3

# Linear with in_features=N, out_features=K
# This expects inputs that are (M, N) in shape

# The weight tensor for this layer is (K, N), since it applies x * W^T + b
layer = nn.Linear(N, K, bias=False)
lseq = torch.tensor(
    [
        [10.0, 20.0, 30.0],
        [100.0, 200.0, 300.0],
        [1000.0, 2000.0, 3000.0],
        [10000.0, 20000.0, 30000.0],
    ]
).T

with torch.no_grad():
    layer.weight.copy_(lseq)

x = torch.linspace(0.1, 2.4, M * N).reshape(M, N)
out = layer(x)
print(f"x: {x}")
print(f"Linear out: {out}")

# Reproducing the same thing with plain Numpy
import numpy as np

W = lseq.numpy()
x_np = x.numpy()
out_np = x_np @ W.T
print(f"Linear out (Numpy): {out_np}")
