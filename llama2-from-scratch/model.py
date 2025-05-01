import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple


def precompute_theta_pos_frequencies(
    head_dim: int, seq_len: int, device: str, theta: float = 10000.0
):
    # As written in the paragraph 3.2.2 of the paper
    # >> In order to generalize our results in 2D to any xi ∈ Rd where **d is even**, [...]
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    # Build the theta parameter
    # According to the formula theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
    # Shape: (Head_Dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # Shape: (Head_Dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)  # (Dim / 2)
    # Construct the positions (the "m" parameter)
    # Shape: (Seq_Len)
    m = torch.arange(seq_len, device=device)
    # Multiply each theta by each position using the outer product.
    # Shape: (Seq_Len) outer_product* (Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs = torch.outer(m, theta).float()
    # We can compute complex numbers in the polar form c = R * exp(m * theta), where R = 1 as follows:
    # (Seq_Len, Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # Separate the last dimension pairs of two values, representing the real and imaginary parts of the complex number
    # Two consecutive values will become a single complex number
    # (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, H, Head_Dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Reshape the freqs_complex tensor to match the shape of the x_complex tensor. So we need to add the batch dimension and the head dimension
    # (Seq_Len, Head_Dim/2) --> (1, Seq_Len, 1, Head_Dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # Multiply each complex number in the x_complex tensor by the corresponding complex number in the freqs_complex tensor
    # Which results in the rotation of the complex number as shown in the Figure 1 of the paper
    # (B, Seq_Len, H, Head_Dim/2) * (1, Seq_Len, 1, Head_Dim/2) = (B, Seq_Len, H, Head_Dim/2)
    x_rotated = x_complex * freqs_complex
    # Convert the complex number back to the real number
    # (B, Seq_Len, H, Head_Dim/2) -> (B, Seq_Len, H, Head_Dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, Seq_Len, H, Head_Dim/2, 2) -> (B, Seq_Len, H, Head_Dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32  # number of heads for queries
    n_kv_heads: Optional[int] = None  # number of heads for keys and values
    vocab_size: int = -1  # set when we load the tokenizer

    # hidden dim of FF layers.
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None

    norm_eps: float = 1e-5

    # Needed for KV cache
    max_batchsize: int = 32
    max_seq_len: int = 2048

    device: str = None


def repeat_kv(x: torch.Tensor, n_rep: int):
    batch_size, seq_len, n_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        # (B, seq_len, n_heads, 1, head_dim)
        return (
            x[:, :, :, None, :]
            .expand(batch_size, seq_len, n_heads, n_rep, head_dim)
            .reshape(batch_size, seq_len, n_heads * n_rep, head_dim)
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # The gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        return self.weight * self._norm(x.float()).type_as(x)


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        # number of key/value heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads

        # number of query heads
        self.n_heads_q = args.n_heads

        # Ratio between the number of heads for queries and keys/values
        # (how many times the k/v heads should be repeated to match the q heads)
        self.n_rep = self.n_heads_q // self.n_kv_heads

        # dimension of each head
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads_q * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros(
            (args.max_batchsize, args.max_seq_len, self.n_kv_heads, self.head_dim),
            device=args.device,
        )
        self.cache_v = torch.zeros(
            (args.max_batchsize, args.max_seq_len, self.n_kv_heads, self.head_dim),
            device=args.device,
        )

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape  # (B, seq_len=1, dim)

        xq = self.wq(x)  # (B, 1, n_heads_q * head_dim)
        xk = self.wk(x)  # (B, 1, n_kv_heads * head_dim)
        xv = self.wv(x)  # (B, 1, n_kv_heads * head_dim)

        xq = xq.view(
            batch_size, seq_len, self.n_heads_q, self.head_dim
        )  # (B, 1, n_heads_q, head_dim)
        xk = xk.view(
            batch_size, seq_len, self.n_kv_heads, self.head_dim
        )  # (B, 1, n_kv_heads, head_dim)
        xv = xv.view(
            batch_size, seq_len, self.n_kv_heads, self.head_dim
        )  # (B, 1, n_kv_heads, head_dim)

        # positional embeddings applied to q and k
        xq = apply_rotary_embeddings(
            xq, freqs_complex, device=x.device
        )  # (B, 1, n_heads_q, head_dim)
        xk = apply_rotary_embeddings(
            xk, freqs_complex, device=x.device
        )  # (B, 1, n_kv_heads, head_dim)

        # Replace the entry in the cache for this token
        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

        # Retrieve all the cached keys and values so far
        # (B, Seq_Len, n_kv_heads, head_dim)
        keys = self.cache_k[:batch_size, : start_pos + seq_len]
        values = self.cache_v[:batch_size, : start_pos + seq_len]

        # Repeat the heads of the K and V to reach the number of heads of the Q
        # (this is not an optimized implementation, but it is the most readable)
        keys = repeat_kv(keys, self.n_rep)  # (B, Seq_Len, n_heads_q, head_dim)
        values = repeat_kv(values, self.n_rep)  # (B, Seq_Len, n_heads_q, head_dim)

        # Reshape to parallelize the heads
        # (B, 1, H_q, head_dim) -> (B, H_q, 1, head_dim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        out = torch.matmul(scores, values)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(out)  # (B, 1, dim)


class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # Normalization before self-attention
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Normalization before feed-forward
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (B, Seq_Len, Dim) -> (B, Seq_Len, Dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert (
            args.vocab_size > 0
        ), "vocab_size must be set before initializing the model"

        self.args = args

        # Names have to align with the stored Llama2 checkpoints so PyTorch
        # knows how to load them
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embedding = nn.Embedding(args.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(
            self.args.dim // self.args.n_heads,
            self.args.max_seq_len * 2,
            device=self.args.device,
        )

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (B, seq_len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "only one token at a time"

        # Embed the tokens
        # (B, seq_len) -> (B, seq_len, dim)
        h = self.tok_embedding(tokens)

        # Retrieve the position embeddings pairs (m, theta) corresponding
        # to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos : start_pos + seq_len]

        # Apply all layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output
