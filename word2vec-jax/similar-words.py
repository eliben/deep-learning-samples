import pickle
import sys
import numpy as np


def find_similar_words(word, vocab, inv_vocab, embedding, top_k=10):
    """Finds the top_k most similar words to a given word.

    Returns a list of (word, similarity) pairs.
    """
    if word not in vocab:
        return []

    word_id = vocab[word]
    word_embedding = embedding[word_id]

    # Compute cosine similarity between the word and all other words using numpy.
    norms = np.linalg.norm(embedding, axis=1)
    similarities = np.dot(embedding, word_embedding) / (norms * np.linalg.norm(word_embedding))

    # Extract the top_k indices with the highest similarities
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    return  [(inv_vocab[word], similarities[word]) for word in top_k_indices]


if __name__ == "__main__":
    # Load model parameters from pickle file passed as argument

    if len(sys.argv) < 3:
        print("Usage: similar-words.py <checkpoint.pickle> <train-data.pickle>")
        sys.exit(1)

    model_params_file = sys.argv[1]
    with open(model_params_file, "rb") as file:
        model_params = pickle.load(file)

    train_data_file = sys.argv[2]
    with open(train_data_file, "rb") as file:
        train_data = pickle.load(file)
        vocab = train_data["vocab"]

    # The projection array is our embedding matrix.
    # Its shape is (V, D).
    embedding = model_params["projection"]

    inv_vocab = {v: k for k, v in vocab.items()}

    # Find similar words to "king".
    word = 'war'
    similar_words = find_similar_words(word, vocab, inv_vocab, embedding)
    print(f"Words similar to '{word}':")
    for word, similarity in similar_words:
        print(f"{word}: {similarity:.2f}")

