# Explore the shapes around tf.reduce_sum
from __future__ import print_function
import numpy as np
import tensorflow as tf

batch_dim = 16
feature_dim = 3

init = np.random.uniform(0.0, 0.1, (batch_dim, feature_dim))

data = tf.constant(init)
# reduce_sum along dimension 0, keeping dim 0 with size 1
reduced_along_0 = tf.reduce_sum(data, reduction_indices=0, keep_dims=True)
# reduce_sum along dimension 1, keeping dim 1 with size 1
reduced_along_1 = tf.reduce_sum(data, reduction_indices=1, keep_dims=True)
# reduce_sum along dimension 1, collapsing dim 1
reduced_along_1_nodim = tf.reduce_sum(data, reduction_indices=1,
                                      keep_dims=False)

print('data:\n', init)

with tf.Session() as sess:
    print('reduced_along_0:', reduced_along_0.get_shape(),
          '\n', sess.run(reduced_along_0))
    print('reduced_along_1:\n', reduced_along_1.get_shape(),
          '\n', sess.run(reduced_along_1))
    print('reduced_along_1_nodim:\n', reduced_along_1_nodim.get_shape(),
          '\n', sess.run(reduced_along_1_nodim))
