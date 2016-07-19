from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf

try:
  from matplotlib import pylab
  HAS_PYLAB = True
except ImportError:
  HAS_PYLAB = False

from six.moves import range
from six.moves.urllib.request import urlretrieve

try:
  from sklearn.manifold import TSNE
  HAS_SKLEARN = Trie
except ImportError:
  HAS_SKLEARN = False

from word_utils import read_data, build_dataset, report_words_distance

filename = 'text8.zip'
words = read_data(filename)
# Only the vocabulary_size most common words are retained in the dictionary. All
# others are mapped to UNK.
vocabulary_size = 50000
data, count, dictionary, reverse_dictionary = build_dataset(words)

print('Total # of words', len(words))
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10])
del words  # Hint to reduce memory.

# State for generate_batch_skipgram.
data_index = 0

def generate_batch_skipgram(batch_size, num_skips, skip_window):
    """Generate a batch of data for training.

    Args:
        batch_size: Number of samples to generate in the batch.
        skip_window:
            How many words to consider around the target word, left and right.
            With skip_window=2, in the sentence above for "consider" we'll
            build the window [words, to, consider, around, the].
        num_skips:
            For skip-gram, we map target word to adjacent words in the window
            around it. This parameter says how many adjacent word mappings to
            add to the batch for each target word. Naturally it can't be more
            than skip_window * 2.

    Returns:
        batch, labels - ndarrays with IDs.
        batch: Row vector of size batch_size containing target words.
        labels:
            Column vector of size batch_size containing a randomly selected
            adjacent word for every target word in 'batch'.
    """
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]

    # buffer is a sliding window through the 'data' list. We initially fill it
    # with the first 'span' IDs in 'data'.
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # Generate the batch data in two nested loops. The outer loop takes the next
    # target word from 'data'; the inner loop picks random 'num_skips' adjacent
    # words to map from the target.
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

print('data:', [reverse_dictionary[di] for di in data[:8]])

for num_skips, skip_window in [(2, 1), (4, 2)]:
    data_index = 0
    batch, labels = generate_batch_skipgram(
        batch_size=8,
        num_skips=num_skips,
        skip_window=skip_window)
    print('\nwith num_skips = %d and skip_window = %d:' %
          (num_skips, skip_window))
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64  # Number of negative examples to sample.


def do_report_distances(emb):
    report_words_distance('apple', 'banana', dictionary, emb)
    report_words_distance('apple', 'fruit', dictionary, emb)
    report_words_distance('apple', 'hebrew', dictionary, emb)
    report_words_distance('apple', 'help', dictionary, emb)
    report_words_distance('apple', 'seven', dictionary, emb)

graph = tf.Graph()

with graph.as_default(), tf.device('/cpu:0'):
    # Input data.
    train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Variables.
    # The embeddings is a VxN matrix, where V is the vocabulary size and N
    # is the embedding dimensionality.
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size
                                                ], -1.0, 1.0))
    softmax_weights = tf.Variable(tf.truncated_normal(
        [vocabulary_size, embedding_size],
        stddev=1.0 / math.sqrt(embedding_size)))
    softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Model.
    # Look up embeddings for inputs.
    embed = tf.nn.embedding_lookup(embeddings, train_dataset)
    # Compute the softmax loss, using a sample of the negative labels each time.
    loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, embed,
                                   train_labels, num_sampled, vocabulary_size))

    # Optimizer.
    # Note: The optimizer will optimize the softmax_weights AND the embeddings.
    # This is because the embeddings are defined as a variable quantity and the
    # optimizer's `minimize` method will by default modify all variable
    # quantities that contribute to the tensor it is passed. See docs on
    # `tf.train.Optimizer.minimize()` for more details.
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

    # Compute the similarity between minibatch examples and all embeddings.
    # We use the cosine distance:
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                              valid_dataset)
    similarity = tf.matmul(valid_embeddings,
                           tf.transpose(normalized_embeddings))

num_steps = 93001

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    initial_embeddings = embeddings.eval()
    do_report_distances(initial_embeddings)
    average_loss = 0
    for step in range(num_steps):
        batch_data, batch_labels = generate_batch_skipgram(batch_size, num_skips,
                                                  skip_window)
        feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += l
        if step % 2000 == 0:
            if step > 0:
                average_loss = average_loss / 2000
            # The average loss is an estimate of the loss over the last 2000
            # batches.
            print('Average loss at step %d: %f' % (step, average_loss))
            average_loss = 0
        # note that this is expensive (~20% slowdown if computed every 500
        # steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log = '%s %s,' % (log, close_word)
                print(log)
    final_embeddings = normalized_embeddings.eval()

print('final_embeddings shape:', final_embeddings.shape)
print('Reporting after training')
do_report_distances(final_embeddings)

num_points = 50

if HAS_SKLEARN:
  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points+1, :])

def plot(embeddings, labels):
  assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
  pylab.figure(figsize=(15,15))  # in inches
  for i, label in enumerate(labels):
    x, y = embeddings[i,:]
    pylab.scatter(x, y)
    pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                   ha='right', va='bottom')
  pylab.show()

words = [reverse_dictionary[i] for i in range(1, num_points+1)]

if HAS_PYLAB:
  plot(two_d_embeddings, words)

