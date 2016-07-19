from __future__ import print_function
import itertools
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
        label = np.zeros((batch_size, 1), dtype=np.int32)
        for b in range(batch_size):
            window_end = (data_index + window_size) % len(data)
            window = data[data_index:window_end]
            context[b, 0:context_size] = window[:context_size]
            context[b, context_size:] = window[context_size + 1:]
            label[b, 0] = window[context_size]
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

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
context_size = 2 # How many words to take for context, left and right

# Number of input words to the network
context_full_size = context_size * 2

# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64  # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default(), tf.device('/cpu:0'):
    # Input data.
    train_dataset = tf.placeholder(tf.int32,
                                   shape=[batch_size, context_full_size])
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
    # Look up embeddings for inputs, for each input...
    # The shape should be (batch_size, context_full_size, embedding_size).
    # We want to average all the context vectors within each batch, so we
    # reduce-mean along dimension 1.
    embed = tf.nn.embedding_lookup(embeddings, train_dataset)
    embed_mean = tf.reduce_mean(embed, reduction_indices=[1])

    # Compute the softmax loss, using a sample of the negative labels each time.
    loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, embed_mean,
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

num_steps = 23001

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    print('embed shape:', embed.get_shape())
    print('embed_mean shape:', embed_mean.get_shape())
    initial_embeddings = embeddings.eval()
    #do_report_distances(initial_embeddings)
    average_loss = 0
    batch_gen = generate_batch_cbow(data, batch_size, context_size)
    for step, batch in itertools.izip(range(num_steps), batch_gen):
        batch_data, batch_labels = batch
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
