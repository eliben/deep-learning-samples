# Utils for word2vec models
from __future__ import print_function

import os, sys
import collections
import scipy.spatial
from six.moves.urllib.request import urlretrieve
import tensorflow as tf
import zipfile


def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    url = 'http://mattmahoney.net/dc/'
    if not os.path.exists(filename):
        filename, _ = urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified %s' % filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + filename +
                        '. Can you get to it with a browser?')
    return filename


filename = maybe_download('text8.zip', 31344016)


def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


def read_data_asstring(filename):
    """Extract the first file enclosed in a zip file as a string"""
    with zipfile.ZipFile(filename) as f:
        for name in f.namelist():
            # weird to 'return' on the first iteration but this is copy-pasted
            # from assignment 6....
            return tf.compat.as_str(f.read(name))


def build_dataset(words, vocabulary_size=50000):
    """Returns:

    data:
        list of the same length as words, with each word replaced by a unique
        numeric ID.
    count:
        counters for the vocabulary_size most common words in 'words'.
    dictionary:
        maps word->ID
    reverse_dictionary:
        maps ID->word. Note that if the Kth word in 'words' is WORD, and the
        Kth ID in 'data' is 42, then reverse_dictionary[42] is WORD.
    """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


def report_words_distance(w1, w2, dictionary, embeddings):
    id1 = dictionary[w1]
    id2 = dictionary[w2]
    v1 = embeddings[id1]
    v2 = embeddings[id2]
    assert v1.shape == v2.shape
    euc = scipy.spatial.distance.euclidean(v1, v2)
    cos = scipy.spatial.distance.cosine(v1, v2)
    print('Distance between %s and %s:' % (w1, w2))
    print('  Euclidean:', euc)
    print('  Cosine:', cos)

