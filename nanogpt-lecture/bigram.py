import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyper parameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2

torch.manual_seed(1337)
device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Our model is character-based; tokens are characters - all unique characters in
# the dataset.
chars = sorted(set(text))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


class BigramLanguageMode(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        C = vocab_size
        # Embedding layer: a table (C, C). For each vocabulary token, it
        # stores a dense vector of size C. C also serves as the embedding
        # depth in this case, to avoid a separate unembedding layer at
        # the end. Each of the C elements in the dense vector represent
        # logics (un-normalized probabilities) of the corresponding character
        # being generated.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx is (B, T) array of indices in the current context
        # This basically predicts the next token from the current token only
        # (bigram).
        logits = self.token_embedding_table(idx)  # (B, T, C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # Reshape logits and targets to a form cross_entropy likes.
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    # Note: this function is written in a general way for educational purposes;
    # it doesn't actually use the history (sequence) to predict the next token,
    # all it uses is the last token.
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the logits (B, T, C)
            logits, _ = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


xb, yb = get_batch("train")

m = BigramLanguageMode(vocab_size)
m = m.to(device)

print("xb shape: ", xb.shape)
print("yb shape: ", yb.shape)

out, loss = m(xb, yb)
print("out shape: ", out.shape)
print("out: ", out)
print("loss: ", loss)

# Starting with a single zero token (newline)
# Untrained model -- generates garbage
idx = torch.zeros((1, 1), dtype=torch.long).to(device)
toks = m.generate(idx, max_new_tokens=100)[0].tolist()
print(decode(toks))


# Estimate loss on both the train and validation sets. The loss is averaged
# over multiple batches to reduce noise.
@torch.no_grad()
def estimate_loss():
    eval_iters = 200
    out = {}
    m.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out


# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = m(xb, yb)

    # clear old gradients
    optimizer.zero_grad(set_to_none=True)

    # compute new gradients from the loss
    loss.backward()

    # optimization step
    optimizer.step()


# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
