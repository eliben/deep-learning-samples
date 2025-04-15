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
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


# Feed-forward NN with ReLU activation and one hidden layer.
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
        # Here's what we'd like to do for each token in the input/batch:
        #
        # for each token x:
        #   # compute the gate layer to create scores for each expert
        #   gate_logits = gate(x)
        #
        #   # select top K expects with highest scores  
        #   top_logits, top_experts = torch.topk(gate_logits, MP.topK)
        #
        #   # Apply softmax between top scores to get weights that sum to 1.
        #   top_logits = softmax(top_logits)
        #
        #   # For each selected expert, apply the expert to the input token and
        #   # multiply by the corresponding weight.
        #   output = 0
        #   for expert_idx in top_experts:
        #      output += top_logits[expert_idx] * experts[expert_idx](x)
        #   return output

        # x is (B, N, dim)
        # gate is (dim, num_experts)

        # Multiply by gate (dim, num_experts), to get (B, N, num_experts). For
        # each token, we get a score per expert.
        gate_logits = self.gate(x)

        # Select top K experts with the highest logits. top_logits is
        # (B, N, topK), and top_experts is the indices of the selected
        # experts (B, N, topK).
        top_logits, top_experts = torch.topk(gate_logits, MP.topK, sorted=False)

        top_logits = torch.nn.functional.softmax(
            top_logits, dim=-1, dtype=torch.float
        ).to(x.dtype)

        out = torch.zeros_like(x)
        for expert_idx, expert in enumerate(self.experts):
            # TODO ...

        return gate_logits


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
