from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn


@dataclass
class ModelParams:
    dim: int
    hidden_dim: int
    num_experts: int
    topk_experts: int


MP = ModelParams(
    dim=512,
    hidden_dim=2048,
    num_experts=8,
    topk_experts=2,
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
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class FF(nn.Module):
    def __init__(self):
        super().__init__()

        self.w1 = nn.Linear(MP.dim, MP.hidden_dim, bias=False)
        self.w2 = nn.Linear(MP.hidden_dim, MP.dim, bias=False)

    def forward(self, x):
        return self.w2(nn.ReLU(self.w1(x)))


class Moe(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.gate = gate

    def forward(self, x):
        gate_logits = self.gate(x)
        # TODO: continue working here...
        return gate_logits


experts = [FF() for _ in range(MP.num_experts)]
gate = nn.Linear(MP.dim, MP.num_experts, bias=False)
model = Moe(experts, gate)

N = 128

print(f"Model parameters: {MP}")
x = torch.randn(N, MP.dim)
out = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {out.shape}")
