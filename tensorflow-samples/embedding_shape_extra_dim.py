# Explores the shapes involved for embedding lookup in the presence of extra
# dimensions.
from __future__ import print_function
import numpy as np
import tensorflow as tf

vocab_size = 10
embedding_dim = 5
batch_size = 2
context_size = 4

# Each "sample" in a batch is a context of context_size words (think input to a
# CBOW training for word2vec). So the shape of the dataset is
# (batch_size, context_size), where each element of this array is an integer
# value in the range [0, vocab_size) -- word ID.
dataset = tf.placeholder(tf.int32, shape=[batch_size, context_size])

# embedding is a lookup table for IDs, mapping them into
# embedding_dim-dimensional vectors.
embedding = tf.Variable(tf.random_uniform(
    [vocab_size, embedding_dim], -1.0, 1.0))

# The embedding_lookup is clever... the docs say:
#   "The returned tensor has shape shape(ids) + shape(params)[1:]."
# By which they mean that the shape of embed_out will be the shape of dataset,
# with an additional dimension appended on the right; and this additional
# dimension is dimension "embedding_dim". So the result we expect will have
# the shape (batch_size, context_size, embedding_dim).
embed_out = tf.nn.embedding_lookup(embedding, dataset)

with tf.Session() as sess:
    tf.initialize_all_variables().run()

    print('dataset shape:', dataset.get_shape())
    print('embedding shape:', embedding.get_shape())
    print('embed_out shape:', embed_out.get_shape())

    indata = np.random.randint(0, vocab_size, size=(batch_size, context_size))
    print('indata:', indata)
    embedding, embed_out = sess.run(
        fetches=[embedding, embed_out],
        feed_dict={dataset: indata})
    print('embedding:\n', embedding)
    print('embed_out:\n', embed_out)
