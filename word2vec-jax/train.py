import jax.nn
import jax.numpy as jnp
import optax
import numpy as np
import pickle


def generate_train_vectors(train_data, vocab, window_size=4, batch_size=128):
    """Generates training vectors from a list of words and vocabulary.

    Generates (target_batch, context_batch) pairs, where target_batch is a
    (batch_size,) array of target word IDs and context_batch is a (batch_size,)
    array of corresponding contexts; each context is an (2*window_size,) array
    of word IDs.

    Stops when it runs out of data. The last batch may have fewer elements than
    batch_size.
    """
    target_batch = []
    context_batch = []

    for i in range(len(train_data)):
        if i + 2 * window_size >= len(train_data):
            break

        # 'i' is the index of the leftmost word in the context.
        target_word = train_data[i + window_size]
        left_context = train_data[i : i + window_size]
        right_context = train_data[i + window_size + 1 : i + 2 * window_size + 1]
        context_words = left_context + right_context

        if target_word not in vocab:
            continue
        target_batch.append(vocab.get(target_word, "<unk>"))
        context_batch.append([vocab.get(word, "<unk>") for word in context_words])

        if len(target_batch) == batch_size:
            yield np.array(target_batch), np.array(context_batch)
            target_batch = []
            context_batch = []


@jax.jit
def word2vec_forward(params, context):
    """ Forward pass of the word2Vec model.
    
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
    logits = word2vec_forward(params, context) # (batch_size, V)

    target_onehot = jax.nn.one_hot(target, logits.shape[1]) # (batch_size, V)
    loss = optax.losses.softmax_cross_entropy(logits, target_onehot).mean()
    return loss


def train(train_data, vocab):
    V = len(vocab)
    D = 200
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 512
    EPOCHS = 15

    initializer = jax.nn.initializers.glorot_uniform()
    params = {
        "projection": initializer(jax.random.PRNGKey(501337), (V, D)),
        "hidden": initializer(jax.random.PRNGKey(501337), (D, V)),
    }

    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(params)

    for epoch in range(EPOCHS):
        print(f'=== Epoch {epoch+1}')
        for n, (target_batch, context_batch) in enumerate(generate_train_vectors(train_data, vocab, batch_size=BATCH_SIZE)):
            # Shuffle the batch.
            indices = np.random.permutation(len(target_batch))
            target_batch = target_batch[indices]
            context_batch = context_batch[indices]
            
            # Compute the loss and gradients; optimize.
            loss, grads = jax.value_and_grad(word2vec_loss)(params, target_batch, context_batch)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            if n > 0 and n % 1000 == 0:
                print(f'Batch {n}, loss: {loss:.2f}')


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
    print('Approximate number of batches:', len(train_data) // 512)

    train(train_data, vocab)
    # cnt = 0
    # for target_batch, context_batch in generate_train_vectors(
    #     train_data, vocab, batch_size=4
    # ):
    #     cnt += 1
        # print("Target batch:", target_batch)
        # print("Context batch:", context_batch)
        # print()

    # print("Number of batches:", cnt)
