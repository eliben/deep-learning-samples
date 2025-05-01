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
            torch.set_default_dtype(torch.cuda.HalfTensor)
        else:
            torch.set_default_dtype(torch.bfloat16)

        model = Transformer(model_args).to(device)

        if load_model:
            del checkpoint["rope.freqs"]  # we calculate these ourselves
            model.load_state_dict(checkpoint, strict=False)
            print(f"State dict loaded in {time.time() - prev_time:.2f} seconds")

        return LLaMA(model, tokenizer, model_args)

    def text_completion(
        self,
        prompts: list[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
    ):
        if max_gen_len is None:
            max_gen_len = self.model_args.max_seq_len - 1

        # convert each prompt into tokens
        prompt_tokens = [
            self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False)
            for prompt in prompts
        ]
        # make sure the batch size is not too large
        batch_size = len(prompt_tokens)
        assert (
            batch_size <= self.model_args.max_batchsize
        ), f"Batch size {batch_size} exceeds max batch size {self.model_args.max_batchsize}"
        max_prompt_len = max(len(tokens) for tokens in prompt_tokens)

        # make sure the prompt length is not larger than the mex seq len
        assert max_prompt_len <= self.model_args.max_seq_len
        total_len = min(self.model_args.max_seq_len, max_prompt_len + max_gen_len)

        pad_id = self.tokenizer.pad_id()
        tokens = torch.full(
            (batch_size, total_len),
            pad_id,
            dtype=torch.long,
            device=self.model_args.device,
        )
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(
                t, dtype=torch.long, device=self.model_args.device
            )
        eos_reached = torch.full((batch_size,), False, dtype=torch.bool)
        print(eos_reached.dtype)
        prompt_tokens_mask = (
            tokens != pad_id
        )  # True if token is a prompt token, False otherwise
        print(prompt_tokens_mask.dtype)

        for cur_pos in tqdm(range(1, total_len), desc="Generating", unit="tokens"):
            with torch.no_grad():
                logits = self.model.forward(tokens[:, cur_pos - 1 : cur_pos], cur_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                # Greedy sampling
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace the token if it is a padding token
            next_token = torch.where(
                prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            # EOS reached only if we found EOS token for a padding position
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id()
            )
            if all(eos_reached):
                break

        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # cut to the eos token, if present
            if self.tokenizer.eos_id() in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id())
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        return (out_tokens, out_text)

    def _sample_top_p(self, probs, p):
        # (B, vocab_size)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        # (B, vocab_size)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        # (B, vocab_size)
        # (Substracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking)
        mask = probs_sum - probs_sort > p
        # Zero out all the probabilities of tokens that are not selected by the Top P
        probs_sort[mask] = 0.0
        # Redistribute the probabilities so that they sum up to 1.
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        # Sample a token (its index) from the top p distribution
        next_token = torch.multinomial(probs_sort, num_samples=1)
        # Get the token position in the vocabulary corresponding to the sampled index
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token


if __name__ == "__main__":
    torch.manual_seed(0)

    allow_cuda = False  # OOM on my machine
    device = "cuda" if torch.cuda.is_available() and allow_cuda else "cpu"

    prompts = [
        "Simply put, the theory of relativity states that",
        "If Google was an italian company founded in Milan, it would",
        "The meaning of life is",
    ]

    model = LLaMA.build(
        checkpoints_dir="llama-2-7b",
        tokenizer_path="tokenizer.model",
        load_model=True,
        max_seq_len=1024,
        max_batchsize=len(prompts),
        device=device,
    )

    # Inference the model
    out_tokens, out_text = model.text_completion(prompts, max_gen_len=64)
    assert len(out_text) == len(prompts)
    for i in range(len(out_text)):
        print(f"Prompt: {prompts[i]}")
        print(f"Output: {out_text[i]}")
        print(f"Tokens: {out_tokens[i]}")
        print()
