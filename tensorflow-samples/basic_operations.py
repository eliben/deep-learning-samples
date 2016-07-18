from __future__ import print_function
import tensorflow as tf

# Basic constant operations - added to the default global graph.
# The value returned by the constructor represents the output
# of the Constant op.
a = tf.constant(2)
b = tf.constant(3)

# Launch the default graph.
with tf.Session() as sess:
    print('a =', sess.run(a))
    print('b =', sess.run(b))
    print('a + b =', sess.run(a + b))

# Basic Operations with variable as graph input
# The value returned by the constructor represents the output
# of the Variable op. (define as input when running session)
# tf Graph input
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
add = tf.add(a, b)

# Launch the default graph.
with tf.Session() as sess:
    print('a + b [from placeholders]', sess.run(add, feed_dict={a: 2, b: 3}))

# Define two constants and a matmul node on them, in a new default graph.
graph = tf.Graph()

with graph.as_default():
    # Matmul of two constants.
    matrix1 = tf.constant([[1, 1, 1], [2, 2, 2]], dtype=tf.float32)
    matrix2 = tf.constant([[2], [3], [4]], dtype=tf.float32)
    product = tf.matmul(matrix1, matrix2)

with tf.Session(graph=graph) as sess:
    # Ask product for its shape (using TF's shape inference) and also run the
    # graph.
    print('product shape =', product.get_shape())
    print('product =', sess.run(product))
