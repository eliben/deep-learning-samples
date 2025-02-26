import torch
import torch.nn as nn
from torch.nn import functional as F

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")


block_size = 256  # what is the maximum context length for predictions?


# Attention head taken from https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
# with masking/dropout removed
class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, C, head_size, do_mask=False):
        super().__init__()
        self.key = nn.Linear(C, head_size, bias=False)
        self.query = nn.Linear(C, head_size, bias=False)
        self.value = nn.Linear(C, head_size, bias=False)
        self.do_mask = do_mask
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
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

        if self.do_mask:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        else:
            wei = F.softmax(wei, dim=-1)  # (B, T, T)
        # wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, C, num_heads, head_size, do_mask=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(C, head_size, do_mask) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(head_size * num_heads, C)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        # out = self.dropout(self.proj(out))
        return out


print("---- Single attention head, B=1 ----")
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

print("---- Single attention head, B=3 ----")
B = 3
T = 4
C = 2
HS = 2

x = torch.linspace(0.1, 5.4, B * T * C).reshape(B, T, C)
print(x)

bhead = Head(C, HS)

kseq = torch.linspace(0.1, 0.8, C * HS).view(bhead.key.weight.shape)
qseq = torch.linspace(1.1, 1.8, C * HS).view(bhead.query.weight.shape)
vseq = torch.linspace(2.1, 2.8, C * HS).view(bhead.value.weight.shape)

with torch.no_grad():
    bhead.key.weight.copy_(kseq)
    bhead.query.weight.copy_(qseq)
    bhead.value.weight.copy_(vseq)

out = bhead(x)
print(out.shape)
print(out)

print("---- Multi-head attention, B=3 ----")
B = 3
T = 4
C = 6
HS = 2
NH = 3

x = torch.linspace(0.1, 8.4, B * T * C).reshape(B, T, C)
print(x)

mhead = MultiHeadAttention(C, NH, HS)

kseqs = [
    torch.linspace(i + 0.1, i + 0.8, C * HS).view(mhead.heads[0].key.weight.shape)
    for i in range(NH)
]

qseqs = [
    torch.linspace(i + 3.1, i + 3.8, C * HS).view(mhead.heads[0].key.weight.shape)
    for i in range(NH)
]

vseqs = [
    torch.linspace(i + 6.1, i + 6.8, C * HS).view(mhead.heads[0].key.weight.shape)
    for i in range(NH)
]

pseq = torch.linspace(9.1, 9.8, C * HS * NH).view(mhead.proj.weight.shape)

with torch.no_grad():
    for i in range(NH):
        mhead.heads[i].key.weight.copy_(kseqs[i])
        mhead.heads[i].query.weight.copy_(qseqs[i])
        mhead.heads[i].value.weight.copy_(vseqs[i])
    mhead.proj.weight.copy_(pseq)

out = mhead(x)
print(out.shape)
print(out)
