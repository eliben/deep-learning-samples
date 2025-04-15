from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelParams:
    dim: int
    hidden_dim: int
    num_experts: int
    topK: int


MP = ModelParams(
    dim=512,
    hidden_dim=2048,
    num_experts=8,
    topK=2,
)


# Feed-forward NN with ReLU activation and one hidden layer.
class FF(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(MP.dim, MP.hidden_dim, bias=False)
        self.w2 = nn.Linear(MP.hidden_dim, MP.dim, bias=False)

    def forward(self, x):
        return self.w2(F.relu(self.w1(x)))


class Moe(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.gate = gate

    def forward(self, x):
        # x is (B, N, dim)
        # Multiply by gate (dim, num_experts), to get (B, N, num_experts). For
        # each token, we get a score per expert.
        gate_scores = self.gate(x)

        # Select top K experts with the highest scores. top_scores is
        # (B, N, topK), and top_experts is the indices of the selected
        # experts (B, N, topK).
        top_scores, top_experts = torch.topk(gate_scores, MP.topK, sorted=False)

        # Apply softmax to the top scores to get weights that sum to 1.
        weights = F.softmax(top_scores, dim=-1)

        out = torch.zeros_like(x)
        # For each token in batch and sequence.
        for b in range(x.shape[0]):
            for n in range(x.shape[1]):
                # Select the top K experts and their corresponding weights for
                # this token.
                for expert_idx, weight in zip(top_experts[b, n], weights[b, n]):
                    # Apply the expert to the input token and multiply by the
                    # corresponding weight.
                    out[b, n] += weight * self.experts[expert_idx](x[b, n])

        return out


experts = [FF() for _ in range(MP.num_experts)]
gate = nn.Linear(MP.dim, MP.num_experts, bias=False)
model = Moe(experts, gate)

B = 16
N = 64

print(f"Model parameters: {MP}")
x = torch.randn(B, N, MP.dim)
out = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {out.shape}")
