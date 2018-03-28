from __future__ import print_function
import numpy as np
import tensorflow as tf

in_channels = 32 # 3 for RGB, 32, 64, 128, ...
out_channels = 64 # 128, 256, ...
ones_3d = np.ones((5,5,in_channels)) # input is 3d, in_channels = 32
# filter must have 3d-shpae x number of filters = 4D
weight_4d = np.ones((3,3,in_channels, out_channels))
strides_2d = [1, 1, 1, 1]

in_3d = tf.constant(ones_3d, dtype=tf.float32)
filter_4d = tf.constant(weight_4d, dtype=tf.float32)

in_width = int(in_3d.shape[0])
in_height = int(in_3d.shape[1])

filter_width = int(filter_4d.shape[0])
filter_height = int(filter_4d.shape[1])

input_3d   = tf.reshape(in_3d, [1, in_height, in_width, in_channels])
kernel_4d = tf.reshape(filter_4d, [filter_height, filter_width, in_channels, out_channels])

#output stacked shape is 3D = 2D x N matrix
output_3d = tf.nn.conv2d(input_3d, kernel_4d, strides=strides_2d, padding='SAME')

with tf.Session() as sess:
    print(sess.run(output_3d))
