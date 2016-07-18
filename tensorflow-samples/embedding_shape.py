# Explores the shapes involved in the embedding_lookup op.
from __future__ import print_function
import numpy as np
import tensorflow as tf

vocab_size = 10
embedding_dim = 5
batch_size = 16

# The data set is an array of integer values, each being an ID in the range
# [0, vocab_size)
dataset = tf.placeholder(tf.int32, shape=[batch_size])

# embedding is a lookup table for IDs, mapping them into
# embedding_dim-dimensional vectors.
embedding = tf.Variable(tf.random_uniform(
    [vocab_size, embedding_dim], -1.0, 1.0))

embed_out = tf.nn.embedding_lookup(embedding, dataset)

with tf.Session() as sess:
    tf.initialize_all_variables().run()

    print('dataset shape:', dataset.get_shape())
    print('embedding shape:', embedding.get_shape())
    print('embed_out shape:', embed_out.get_shape())

    indata = np.random.randint(0, vocab_size, size=(batch_size,))
    print('indata:', indata)
    embedding, embed_out = sess.run(
        fetches=[embedding, embed_out],
        feed_dict={dataset: indata})
    print('embedding:\n', embedding)
    print('embed_out:\n', embed_out)
