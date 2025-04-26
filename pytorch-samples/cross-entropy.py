import torch
import torch.nn.functional as F

# Suppose we have 2 samples and 3 classes
# logits can be any real numbers (raw, un-normalized scores)
logits1 = torch.tensor([[2.0, 0.5, 0.1], [0.1, 0.2, 3.0]])  # sample 0  # sample 1

# target labels: each in {0,1,2}, shape (batch,)
targets = torch.tensor([0, 2])

# compute loss (default reduction='mean')
loss1 = F.cross_entropy(logits1, targets)
print(f"Mean loss1: {loss1.item():.3f}")

logits2 = torch.tensor([[1.0, 1.5, 1.1], [2.1, 2.2, 2.0]])  # sample 0  # sample 1
loss2 = F.cross_entropy(logits2, targets)
print(f"Mean loss2: {loss2.item():.3f}")

# If you want per-sample losses:
loss_none = F.cross_entropy(logits1, targets, reduction="none")
print(f"Per-sample losses: {loss_none.tolist()}")
