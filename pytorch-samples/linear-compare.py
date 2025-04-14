import torch
import torch.nn as nn

# input shape is (M, N)
# output shape is (M, K)
M = 6
N = 4
K = 3

# Linear with in_features=N, out_features=K
# This expects inputs that are (M, N) in shape, and then
# multiplies (M, N) by (N, K) to get (M, K).

# Internally, nn.Linear creates its tensor as (out_features, in_features),
# so when we assign to its weights directly we have to use the same shape,
# which is (K, N) in our case.
layer = nn.Linear(N, K, bias=False)
lseq = torch.tensor(
    [
        [10.0, 100.0, 1000.0, 10000.0],
        [20.0, 200.0, 2000.0, 20000.0],
        [30.0, 300.0, 3000.0, 30000.0],
    ]
)

with torch.no_grad():
    layer.weight.copy_(lseq)

x = torch.linspace(0.1, 2.4, M * N).reshape(M, N)
out = layer(x)
print(f"x:\n {x}")
print(f"Linear out:\n {out}")

# Reproducing the same thing with plain Numpy
import numpy as np

W = lseq.numpy()
x_np = x.numpy()
out_np = x_np @ W.T
print(f"Linear out (Numpy):\n {out_np}")
