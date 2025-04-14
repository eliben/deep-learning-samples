# Experimenting with Pytorch's topk function
import torch
import torch.nn as nn

# New torch array with given values
x = torch.tensor([10, 20, 11, 4, 8, 19])

values, indices = torch.topk(x, 2)
print(f"Top 2 values: {values}")
print(f"Top 2 indices: {indices}")
