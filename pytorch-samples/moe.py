from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn


@dataclass
class ModelParams:
    dim: int
    hidden_dim: int
    num_experts: int
    num_layers: int
    dropout: float


MP = ModelParams(
    dim=512,
    hidden_dim=2048,
    num_experts=8,
    topk_experts=2,
)

# TODO: annotate dimensions here: in the general case, use (B, N, D) notation


# TODO: figure this out, follows https://arxiv.org/pdf/2002.05202
# "GLU variants improve transformer"
class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

        self.w1 = nn.Linear(MP.dim, MP.hidden_dim, bias=False)
        self.w2 = nn.Linear(MP.hidden_dim, MP.dim, bias=False)
        self.w3 = nn.Linear(MP.dim, MP.hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class MyModel(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.gate = gate

    def forward(self, x):
        gate_logits = self.gate(x)
        return self.linear(x)


model = MyModel()
