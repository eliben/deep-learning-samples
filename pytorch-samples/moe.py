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

# TODO: annotate dimensions here: in the general case, use (B, N, D) notation


# SwiGLU from the paper "GLU variants improve transformer"
# [https://arxiv.org/pdf/2002.05202]
class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

        self.w1 = nn.Linear(MP.dim, MP.hidden_dim, bias=False)
        self.w2 = nn.Linear(MP.hidden_dim, MP.dim, bias=False)
        self.w3 = nn.Linear(MP.dim, MP.hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


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
        # gate is (dim, num_experts)

        # Multiply by gate (dim, num_experts), to get (B, N, num_experts). For
        # each token, we get a score per expert.
        gate_logits = self.gate(x)

        # Select top K experts with the highest logits. top_logits is
        # (B, N, topK), and top_experts is the indices of the selected
        # experts (B, N, topK).
        top_logits, top_experts = torch.topk(gate_logits, MP.topK, sorted=False)

        weights = F.softmax(top_logits, dim=-1, dtype=torch.float).to(x.dtype)

        out = torch.zeros_like(x)
        for b in range(x.shape[0]):
            for n in range(x.shape[1]):
                # Select the top K experts and their corresponding weights for
                # this token.
                selected_experts = top_experts[b, n]
                selected_weights = top_logits[b, n]

                for expect_idx, weight in zip(top_experts[b, n], weights[b, n]):
                    # Apply the expert to the input token and multiply by the
                    # corresponding weight.
                    out[b, n] += weight * self.experts[expect_idx](x[b, n])

        return out


experts = [FF() for _ in range(MP.num_experts)]
gate = nn.Linear(MP.dim, MP.num_experts, bias=False)
model = Moe(experts, gate)

B = 16
N = 128

print(f"Model parameters: {MP}")
x = torch.randn(B, N, MP.dim)
out = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {out.shape}")
