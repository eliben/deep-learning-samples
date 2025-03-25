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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find similar words using word embeddings."
    )
    parser.add_argument("word", help="The word to find similar words for.")
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

    similar_words = find_similar_words(args.word, vocab, inv_vocab, embedding)
    print(f"Words similar to '{args.word}':")
    for word, similarity in similar_words:
        print(f"{word}: {similarity:.2f}")
