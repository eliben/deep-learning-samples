# Takes the raw dataset (text8) and creates training data from it, by
# subsampling frequent words and creating a vocabulary. The training data is
# saved to a pickle file as the dict
# {
#     "train_data": list of words,
#     "vocab": dict from word to ID
# }
#
from collections import Counter
import random
import math
import pickle


def read_words_from_file(file_path):
    """Reads whitespace-separated words from a file.

    Returns a list of words.
    """
    with open(file_path, "r") as file:
        return file.read().split()


def subsample(words, threshold=1e-4):
    """Subsample frequent words, return a new list of words.

    Follows the subsampling procedure described in the paper "Distributed
    Representations of Words and Phrases and their Compositionality" by
    Mikolov et al. (2013).
    """
    word_counts = Counter(words)
    total_count = len(words)
    freqs = {word: count / total_count for word, count in word_counts.items()}

    # Common words (freq(word) > threshold) are kept with a computed
    # probability, while rare words are always kept.
    p_keep = {
        word: math.sqrt(threshold / freqs[word]) if freqs[word] > threshold else 1
        for word in word_counts
    }
    return [word for word in words if random.random() < p_keep[word]]


def make_vocabulary(words, top_k=20000):
    """Creates a vocabulary from a list of words.

    Keeps the top_k most common words and assigns an index to each word. The
    index 0 is reserved for the "<unk>" token.
    """
    word_counts = Counter(words)
    vocab = {"<unk>": 0}
    for word, _ in word_counts.most_common(top_k - 1):
        vocab[word] = len(vocab)
    return vocab


if __name__ == "__main__":
    input_filename = "text8"
    print("Reading words from", input_filename)
    words = read_words_from_file(input_filename)
    print("Number of words:", len(words))

    ss = subsample(words)
    print("Number of words after subsampling:", len(ss))

    vocab = make_vocabulary(ss)
    print("Vocabulary size:", len(vocab))

    # TODO: this should be IDs... in a function
    train_data = [word for word in ss if word in vocab]
    print("Number of words in training data:", len(train_data))

    output_filename = "train-data.pickle"
    print("Saving training data to", output_filename)
    with open(output_filename, "wb") as file:
        pickle.dump({"train_data": train_data, "vocab": vocab}, file)
