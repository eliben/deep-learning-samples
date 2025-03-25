import jax.nn
import jax.numpy as jnp
import optax
import numpy as np
import pickle


def generate_train_vectors(train_data, vocab, window_size=4, batch_size=128):
    """Generates training vectors from a list of words and vocabulary.

    Generates (target_batch, context_batch) pairs, where target_batch is a
    (batch_size,) array of target word IDs and context_batch is a
    (batch_size, 2*window_size) array of context word IDs.

    Stops when it runs out of data. The leftover data (whatever doesn't fit in
    the last batch) will be discared.
    """
    target_batch = np.zeros(batch_size, dtype=np.int32)
    context_batch = np.zeros((batch_size, 2 * window_size), dtype=np.int32)
    batch_idx = 0

    for i in range(len(train_data)):
        if i + 2 * window_size >= len(train_data):
            break

        # 'i' is the index of the leftmost word in the context.
        target_word = train_data[i + window_size]
        left_context = train_data[i : i + window_size]
        right_context = train_data[i + window_size + 1 : i + 2 * window_size + 1]

        target_batch[batch_idx] = vocab.get(target_word, 0)
        context_batch[batch_idx, :] = np.array(
            [vocab.get(word, 0) for word in left_context + right_context]
        )

        batch_idx += 1
        if batch_idx == batch_size:
            yield np.array(target_batch), np.array(context_batch)
            batch_idx = 0


@jax.jit
def word2vec_forward(params, context):
    """Forward pass of the word2Vec model.

    context is a (batch_size, 2*window_size) array of word IDs.

    V is the vocabulary size, D is the embedding dimension.
    """
    # Indexing into (V, D) matrix with a batch of IDs. The output shape
    # is (batch_size, 2*window_size, D).
    projection = params["projection"][context]

    # Compute average across the context word. The output shape is
    # (batch_size, D).
    avg_projection = jnp.mean(projection, axis=1)

    # (batch_size, D) @ (D, V) -> (batch_size, V)
    hidden = jnp.dot(avg_projection, params["hidden"])
    return hidden


@jax.jit
def word2vec_loss(params, target, context):
    """Compute the loss of the word2Vec model."""
    logits = word2vec_forward(params, context)  # (batch_size, V)

    target_onehot = jax.nn.one_hot(target, logits.shape[1])  # (batch_size, V)
    loss = optax.losses.softmax_cross_entropy(logits, target_onehot).mean()
    return loss


def train(train_data, vocab):
    V = len(vocab)
    D = 200
    LEARNING_RATE = 1e-3
    WINDOW_SIZE = 8
    BATCH_SIZE = 1024
    EPOCHS = 25

    initializer = jax.nn.initializers.glorot_uniform()
    params = {
        "projection": initializer(jax.random.PRNGKey(501337), (V, D)),
        "hidden": initializer(jax.random.PRNGKey(501337), (D, V)),
    }

    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(params)

    print("Approximate number of batches:", len(train_data) // BATCH_SIZE)

    for epoch in range(EPOCHS):
        print(f"=== Epoch {epoch + 1}")
        epoch_loss = []
        for n, (target_batch, context_batch) in enumerate(
            generate_train_vectors(
                train_data, vocab, window_size=WINDOW_SIZE, batch_size=BATCH_SIZE
            )
        ):
            # Shuffle the batch.
            indices = np.random.permutation(len(target_batch))
            target_batch = target_batch[indices]
            context_batch = context_batch[indices]

            # Compute the loss and gradients; optimize.
            loss, grads = jax.value_and_grad(word2vec_loss)(
                params, target_batch, context_batch
            )
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            epoch_loss.append(loss)
            if n > 0 and n % 1000 == 0:
                print(f"Batch {n}")

        print(f"Epoch loss: {np.mean(epoch_loss):.2f}")
        checkpoint_filename = f"checkpoint-{epoch:03}.pickle"
        print("Saving checkpoint to", checkpoint_filename)
        with open(checkpoint_filename, "wb") as file:
            pickle.dump(params, file)


if __name__ == "__main__":
    print(jax.devices())
    print(jax.default_backend())
    # Load train-data from pickle file
    with open("train-data.pickle", "rb") as file:
        data = pickle.load(file)
        train_data = data["train_data"]
        vocab = data["vocab"]
    print("Number of words in train data:", len(train_data))
    print("Vocabulary size:", len(vocab))
    print(
        "First 10 words and their IDs:",
        [(word, vocab[word]) for word in train_data[:10]],
    )

    train(train_data, vocab)
