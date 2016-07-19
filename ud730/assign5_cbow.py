from __future__ import print_function
import math
import numpy as np
import os
import random
import tensorflow as tf

from six.moves import cPickle as pickle

from timer import Timer
from word_utils import read_data, build_dataset, report_words_distance

def generate_batch_cbow(data, batch_size, context_size):
    """
    Args:
        data: List of IDs - the input sequence.
        batch_size: Number of samples to generate.
        context_size:
            How many words to consider around the target word, left and right.
            With context_size=2, in the sentence above for "consider" as the
            target word, the context will be [words, to, around, the].

    Yields:
        Pairs of (context, label) where context is an array with shape
        (batch_size, context_size * 2) and label is an array with shape
        (batch_size,). For each context vector, a single label is matched
        (target ID).
    """
    data_index = 0
    window_size = 2 * context_size + 1
    while True:
        context = np.zeros((batch_size, context_size * 2), dtype=np.int32)
        label = np.zeros((batch_size,), dtype=np.int32)
        for b in range(batch_size):
            window_end = (data_index + window_size) % len(data)
            window = data[data_index:window_end]
            context[b, 0:context_size] = window[:context_size]
            context[b, context_size:] = window[context_size + 1:]
            label[b] = window[context_size]
            data_index = (data_index + 1) % len(data)
        yield (context, label)

pickle_filename = 'textdata.pickle'
# Only the vocabulary_size most common words are retained in the dictionary.
# All others are mapped to UNK.
vocabulary_size = 50000

try:
    with Timer('Loading pickle...'):
        with open(pickle_filename, 'rb') as pickle_file:
            save = pickle.load(pickle_file)
            data = save['data']
            count = save['count']
            dictionary = save['dictionary']
            reverse_dictionary = save['reverse_dictionary']
except:
    print('No pickle... recomputing data.')
    filename = 'text8.zip'
    with Timer('read_data'):
        words = read_data(filename)
    with Timer('build_dataset'):
        data, count, dictionary, reverse_dictionary = build_dataset(words)
    save = {
        'data': data,
        'count': count,
        'dictionary': dictionary,
        'reverse_dictionary': reverse_dictionary,
    }
    with open(pickle_filename, 'wb') as pickle_file:
        pickle.dump(save, pickle_file, pickle.HIGHEST_PROTOCOL)

print('First words in data:')
print(data[:50])

gen = generate_batch_cbow(data, 10, 2)
for i in range(5):
    print(gen.next())
