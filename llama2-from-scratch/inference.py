from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import ModelArgs, Transformer


class LLaMA:
    def __init__(
        self,
        model: Transformer,
        tokenizer: SentencePieceProcessor,
        model_args: ModelArgs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_args = model_args

    @staticmethod
    def build(
        checkpoints_dir: str,
        tokenizer_path: str,
        load_model: bool,
        max_seq_len: int,
        max_batchsize: int,
        device: str,
    ):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, f"No checkpoints found in {checkpoints_dir}"
            chk_path = checkpoints[0]
            print(f"Loading model from {chk_path}")
            checkpoint = torch.load(chk_path, weights_only=False, map_location="cpu")
            print(f"Model loaded in {time.time() - prev_time:.2f} seconds")
            prev_time = time.time()

        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.load(f)

        model_args = ModelArgs(
            max_seq_len=max_seq_len,
            max_batchsize=max_batchsize,
            device=device,
            **params,
        )

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        model = Transformer(model_args).to(device)

        if load_model:
            del checkpoint["rope.freqs"]  # we calculate these ourselves
            model.load_state_dict(checkpoint, strict=False)
            print(f"State dict loaded in {time.time() - prev_time:.2f} seconds")

        return LLaMA(model, tokenizer, model_args)


if __name__ == "__main__":
    torch.manual_seed(0)

    allow_cuda = False  # OOM on my machine
    device = "cuda" if torch.cuda.is_available() and allow_cuda else "cpu"

    model = LLaMA.build(
        checkpoints_dir="llama-2-7b",
        tokenizer_path="tokenizer.model",
        load_model=True,
        max_seq_len=1024,
        max_batchsize=3,
        device=device,
    )

    print("all OK")
