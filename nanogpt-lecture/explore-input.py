# This script explores the input data and prepares it for training a language
# model.

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("length of dataset is", len(text))
print(text[:200])

# Our model is character-based; tokens are characters - all unique characters in
# the dataset.
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

# Split the data into train (90%) and validation (10%) sets
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print("shapes of train and val: ", train_data.shape, val_data.shape)

# This is the sequence length used for training.
# This snippet demonstrates the training examples taken from the single chunk
# of block_size
# It has multiple examples in it: predict the second character given the first,
# predict the third character given the first and the second, and so on.
block_size = 8
print(train_data[: block_size + 1])

x = train_data[:block_size]
y = train_data[1 : block_size + 1]
for t in range(block_size):
    context = x[: t + 1]
    target = y[t]
    print(f"when input is {context} the target is {target}")


# For efficiency, training is done in batches. Each batch is a list of input
# sequences and the corresponding targets.
# target[b][i] is the target for input[b][:i + 1], as before.
torch.manual_seed(1337)
batch_size = 4
block_size = 8


def get_batch(split):
    data = train_data if split == "train" else val_data
    # random starting indices for the batch
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


xb, yb = get_batch("train")
print("input batch shape: ", xb.shape)
print("inputs:\n", xb)

print("target batch shape: ", yb.shape)
print("targets:\n", yb)

print("----")
for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, : t + 1]
        target = yb[b, t]
        print(f"when input is {context} the target is {target}")
