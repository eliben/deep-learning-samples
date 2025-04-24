with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("length of dataset is", len(text))

# print first 200 characters
print(text[:200])

# All unique characters in the dataset
chars = sorted(set(text))
vocab_size = len(chars)
print("all unique characters:", "".join(chars))
print("vocab size:", vocab_size)

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string

print(encode("hii there"))
print(decode(encode("hii there")))

# Encoode the entire text dataset into a torch.Tensor
import torch

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:200])
