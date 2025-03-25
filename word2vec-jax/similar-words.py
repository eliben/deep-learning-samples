import pickle
import numpy as np
import argparse
from itertools import islice


def find_similar_words(word, vocab, inv_vocab, embedding, top_k=10):
    """Finds the top_k most similar words to a given word.

    Returns a list of (word, similarity) pairs.
    """
    if word not in vocab:
        return []

    word_id = vocab[word]

    # embedding is of shape (V, D), where V is the vocabulary size and D is the
    # embedding dimension.
    word_embedding = embedding[word_id]  # (D,)

    # Compute cosine similarity between the word and all other words using
    # numpy. The result is a (V,) array of cosine similarities of word with each
    # of the vocabulary words.
    norms = np.linalg.norm(embedding, axis=1)  # (V,)
    similarities = np.dot(embedding, word_embedding) / (
        norms * np.linalg.norm(word_embedding)
    )

    # Extract the top_k indices with the highest similarities
    top_k_words = np.argsort(similarities)
    return [
        (inv_vocab[word], similarities[word])
        for word in islice(reversed(top_k_words), top_k)
    ]


def find_analogies(a, b, c, vocab, inv_vocab, embedding, top_k=10):
    """Finds analogies for "A is to B as C is to ?".

    Returns a list of (word, similarity) pairs.
    """
    if a not in vocab or b not in vocab or c not in vocab:
        return []

    # embedding is of shape (V, D), where V is the vocabulary size and D is the
    # embedding dimension.
    a_embedding = embedding[vocab[a]]  # (D,)
    b_embedding = embedding[vocab[b]]  # (D,)
    c_embedding = embedding[vocab[c]]  # (D,)

    # Compute the analogy vector
    analogy = b_embedding - a_embedding + c_embedding

    # Compute cosine similarity between the analogy and all other words using
    # numpy. The result is a (V,) array of cosine similarities of word with each
    # of the vocabulary words.
    norms = np.linalg.norm(embedding, axis=1)  # (V,)
    similarities = np.dot(embedding, analogy) / (norms * np.linalg.norm(analogy))

    # Extract the top_k indices with the highest similarities
    top_k_words = np.argsort(similarities)
    return [
        (inv_vocab[word], similarities[word])
        for word in islice(reversed(top_k_words), top_k)
    ]


DESCRIPTION = """
Find similar words or analogies using word embeddings. Only one of -analogy
and -word should be specified; if both are specified, -analogy will be used.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "-analogy",
        help="Comma-separated list of words to perform analogy task (e.g., 'king,man,queen').",
    )
    parser.add_argument("-word", help="The word to find similar words for.")
    parser.add_argument(
        "-checkpoint", required=True, help="Path to the checkpoint pickle file."
    )
    parser.add_argument(
        "-traindata", required=True, help="Path to the training data pickle file."
    )
    args = parser.parse_args()

    model_params_file = args.checkpoint
    train_data_file = args.traindata

    with open(model_params_file, "rb") as file:
        model_params = pickle.load(file)
    with open(train_data_file, "rb") as file:
        train_data = pickle.load(file)
        vocab = train_data["vocab"]
        inv_vocab = {v: k for k, v in vocab.items()}

    # The projection array is our embedding matrix.
    # Its shape is (V, D).
    embedding = model_params["projection"]

    if args.analogy:
        a, b, c = args.analogy.split(",")
        analogies = find_analogies(a, b, c, vocab, inv_vocab, embedding)
        print(f"Analogies for '{a} is to {b} as {c} is to ?':")
        for word, similarity in analogies:
            print(f"{word:15} {similarity:.2f}")
    else:
        similar_words = find_similar_words(args.word, vocab, inv_vocab, embedding)
        print(f"Words similar to '{args.word}':")
        for word, similarity in similar_words:
            print(f"{word:15} {similarity:.2f}")
