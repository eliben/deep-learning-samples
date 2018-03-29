from __future__ import print_function
import numpy as np
import numpy.testing as npt
import unittest

# Tensorflow is used to verify the results of numpy computations - you can
# remove its usage if you don't need the testing.
import tensorflow as tf


def conv2d_single_channel(input, w):
    boundary_width = w.shape[0] // 2
    padded_input = np.pad(input,
                          pad_width=boundary_width,
                          mode='constant',
                          constant_values=0)
    output = np.zeros_like(input)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            for fi in range(w.shape[0]):
                for fj in range(w.shape[1]):
                    output[i, j] += padded_input[i + fi, j + fj] * w[fi, fj]
    return output


def tf_conv2d_single_channel(input, w):
    # We only have one item in our "batch", one input channel and one output
    # channel; prepare the shapes TF expects for the conv2d op.
    input_4d = tf.reshape(tf.constant(input, dtype=tf.float32),
                         [1, input.shape[0], input.shape[1], 1])
    kernel_4d = tf.reshape(tf.constant(w, dtype=tf.float32),
                           [w.shape[0], w.shape[1], 1, 1])
    output = tf.nn.conv2d(input_4d, kernel_4d,
                          strides=[1, 1, 1, 1], padding='SAME')
    with tf.Session() as sess:
        return sess.run(output)


class TestConvs(unittest.TestCase):
    def test_single_channel(self):
        inp = np.linspace(-5, 5, 36).reshape(6, 6)
        w = np.linspace(0, 8, 9)[::-1].reshape(3, 3)

        np_ans = conv2d_single_channel(inp, w)
        tf_ans = tf_conv2d_single_channel(inp, w).reshape(6, 6)
        npt.assert_almost_equal(np_ans, tf_ans, decimal=4)


if __name__ == '__main__':
    #unittest.main()

    inp = np.ones((6, 6))
    w = np.zeros((3, 3))
    w[0, 0] = 1
    w[1, 1] = 1
    w[2, 2] = 1

    out = conv2d_single_channel(inp, w)
    print(out)

    #outtf = tf_conf2d_single_channel(inp, w)
    #print(outtf[0, :, :, 0])
