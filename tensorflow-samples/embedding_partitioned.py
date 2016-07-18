# Explores a partitioned embedding.
from __future__ import print_function
import numpy as np
import tensorflow as tf

# The vocabulary is partitioned to npartitions embedding tables using the 'div'
# strategy. This means:
# the first partition holds mappings for [0, vocab_per_partition),
# the second holds [vocab_per_partition, 2 * vocab_per_partition), and so on.
# Each partition is just a regular matrix, so indices have to be computed
# properly. In the second partition, row 0 is the embedding mapping for ID
# number 'vocab_per_partition', etc.
vocab_size = 20
npartitions = 4
vocab_per_partition = vocab_size // npartitions
embedding_dim = 3
batch_size = 16

dataset = tf.placeholder(tf.int32, shape=[batch_size])

embeddings = []
for _ in range(npartitions):
    embeddings.append(tf.Variable(tf.random_uniform(
        [vocab_per_partition, embedding_dim], -1.0, 1.0)))

embed_out = tf.nn.embedding_lookup(embeddings,
                                   dataset,
                                   partition_strategy='div',
                                   validate_indices=True)

with tf.Session() as sess:
    tf.initialize_all_variables().run()

    print('dataset shape:', dataset.get_shape())
    print('embeddings shapes:',
        ["{0}: {1}".format(i, embeddings[i].get_shape())
         for i in range(npartitions)])
    print('embed_out shape:', embed_out.get_shape())

    indata = np.random.randint(0, vocab_size, size=(batch_size,))
    print('indata:', indata)
    emb = [0] * npartitions
    emb[0], emb[1], emb[2], emb[3], embed_out = sess.run(
        fetches=[embeddings[0], embeddings[1], embeddings[2], embeddings[3],
                 embed_out],
        feed_dict={dataset: indata})
    for i in range(npartitions):
        print('embeddings[{0}]\n'.format(i), emb[i])
    print('embed_out:\n', embed_out)
