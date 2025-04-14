# Experimenting with Pytorch's topk function
import torch
import torch.nn as nn

# New torch array with given values
x = torch.tensor([10, 20, 11, 4, 8, 19])

values, indices = torch.topk(x, 2)
print(f"Top 2 values: {values}")
print(f"Top 2 indices: {indices}")


# Numpy equivalent
import numpy as np


def topk_np(x, k):
    # Using np.argpartition to get the indices of the top k elements: they
    # are not necessarily sorted. Can be sorted if needed.
    idx = np.argpartition(x, -k)[-k:]
    return x[idx], idx


x_np = x.numpy()
values_np, indices_np = topk_np(x_np, 2)
print(f"Top 2 values (Numpy): {values_np}")
print(f"Top 2 indices (Numpy): {indices_np}")
