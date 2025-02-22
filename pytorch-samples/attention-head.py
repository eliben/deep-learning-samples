import torch
import torch.nn as nn
from torch.nn import functional as F


# Attention head taken from https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
# with masking/dropout removed
class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, C, head_size):
        super().__init__()
        self.key = nn.Linear(C, head_size, bias=False)
        self.query = nn.Linear(C, head_size, bias=False)
        self.value = nn.Linear(C, head_size, bias=False)
        # self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        # wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        # wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


B = 1
T = 6
C = 4
HS = 4

x = torch.linspace(0.1, 2.4, T * C).reshape(B, T, C)
print(x)

head = Head(C, HS)

kseq = torch.linspace(0.1, 0.4, B * C * HS).view(head.key.weight.shape)
qseq = torch.linspace(1.1, 1.4, B * C * HS).view(head.query.weight.shape)
vseq = torch.linspace(2.1, 2.4, B * C * HS).view(head.value.weight.shape)

with torch.no_grad():
    head.key.weight.copy_(kseq)
    head.query.weight.copy_(qseq)
    head.value.weight.copy_(vseq)

out = head(x)

print(out.shape)
print(out)
# x = torch.randn(2, 10)
# out = model(x)
# print(out)
