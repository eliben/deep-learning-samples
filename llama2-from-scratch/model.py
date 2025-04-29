import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32  # number of heads for queries
    n_kv_heads: Optional[int] = None # number of heads for keys and values
    vocab_size: int = -1 # set when we load the tokenizer
    
    # hidden dim of FF layers.
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None

    norm_eps: float = 1e-5

    # Needed for KV cache
    max_batchsize: int = 32
    max_seq_len: int = 2048

    device: str = None


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size > 0, "vocab_size must be set before initializing the model"

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

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads,
                                                              self.args.max_seq_len* 2,
                                                              device=self.args.device)
        
    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (B, seq_len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "oly one token at a time"

        # Embed the tokens
        # (B, seq_len) -> (B, seq_len, dim)
        h = self.tok_embedding(tokens)

        # Retrieve the position embeddings pairs (m, theta) corresponding
        # to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        # Apply all layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output


