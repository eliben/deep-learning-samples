from collections import Counter
import random
import math

def read_words_from_file(file_path):
    with open(file_path, 'r') as file:
        return file.read().split()


def subsample(words, threshold=1e-4):
    word_counts = Counter(words)
    total_count = len(words)
    freqs = {word: count / total_count for word, count in word_counts.items()}
    p_keep = {word: math.sqrt(threshold / freqs[word]) 
              if freqs[word] > threshold else 1
              for word in word_counts}
    return [word for word in words if random.random() < p_keep[word]]

def make_vocabulary(words, top_k=20000):
    word_counts = Counter(words)
    vocab = {'unk': 0}
    for word, _ in word_counts.most_common(top_k-1):
        vocab[word] = len(vocab)
    return vocab

if __name__ == '__main__':
    words = read_words_from_file('text8')
    print('Number of words:', len(words))

    ss = subsample(words)
    print('Number of words after subsampling:', len(ss))

    vocab = make_vocabulary(ss)
    print('Vocabulary size:', len(vocab))