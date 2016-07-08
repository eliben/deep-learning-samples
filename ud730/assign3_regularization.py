from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

def run_sgd_with_reg():
    # Run stochastic gradient descent, where training samples are fed as
    # minibatches into a TF placeholder at each iteration.
    batch_size = 128

    graph = tf.Graph()
    with graph.as_default():

      # Input data. For the training data, we use a placeholder that will be fed
      # at run time with a training minibatch.
      tf_train_dataset = tf.placeholder(tf.float32,
                                        shape=(batch_size,
                                               image_size * image_size))
      tf_train_labels = tf.placeholder(tf.float32,
                                       shape=(batch_size, num_labels))
      tf_valid_dataset = tf.constant(valid_dataset)
      tf_test_dataset = tf.constant(test_dataset)

      # Regularization factor. 0.01 improves the test set accuracy. 0.005
      # improves it even a bit more.
      # 0.1 makes it worse than no-regularization, so it must be too large
      # (underfitting).
      beta = tf.constant(0.005)

      # Variables.
      weights = tf.Variable(
        tf.truncated_normal([image_size * image_size, num_labels]))
      biases = tf.Variable(tf.zeros([num_labels]))

      # Training computation.
      logits = tf.matmul(tf_train_dataset, weights) + biases
      loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

      reg_loss = loss + beta * tf.nn.l2_loss(weights)

      # Optimizer.
      optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(reg_loss)

      # Predictions for the training, validation, and test data.
      train_prediction = tf.nn.softmax(logits)
      valid_prediction = tf.nn.softmax(
        tf.matmul(tf_valid_dataset, weights) + biases)
      test_prediction = tf.nn.softmax(
              tf.matmul(tf_test_dataset, weights) + biases)

    num_steps = 3001

    with tf.Session(graph=graph) as session:
      tf.initialize_all_variables().run()
      print("Initialized")
      for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
          [optimizer, reg_loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
          print("Minibatch loss at step %d: %f" % (step, l))
          print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
          print("Validation accuracy: %.1f%%" % accuracy(
            valid_prediction.eval(), valid_labels))
      print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


def run_sgd_with_hidden_layer_with_reg(num_steps=3001,
                                       learning_rate=0.1,
                                       l2_reg_beta=0.0,
                                       dropout_keep_prob=0.5):
    """Run SGD with a hidden layer and regularization.

    num_steps:
        Number of steps to run SGD minibatch training. In each step a new
        minibatch is fed through the network.

    learning_rate:
        SGD learning rate, passed direction to GradientDescentOptimizer.

    l2_reg_beta:
        The loss is regularized with L2 loss on all weights. This is the
        regularization coefficient. Setting it to 0 means no L2 loss
        regularization. Otherwise, small values work well.

    dropout_keep_prob:
        Dropout "keep" probability for outputs of the hidden layer. Set to 1 to
        avoid dropout altogether.
    """
    # The "problem" part of the assignment
    batch_size = 128
    nfeatures = image_size * image_size
    nhidden = 1024

    graph = tf.Graph()
    with graph.as_default():
      # Input data. For the training data, we use a placeholder that will be fed
      # at run time with a training minibatch.
      tf_train_dataset = tf.placeholder(tf.float32,
                                        shape=(batch_size, nfeatures))
      tf_train_labels = tf.placeholder(tf.float32,
                                       shape=(batch_size, num_labels))
      tf_dropout_keep_prob = tf.placeholder(tf.float32)
      tf_valid_dataset = tf.constant(valid_dataset)
      tf_test_dataset = tf.constant(test_dataset)

      # setting beta=0 means no L2-loss regularization
      beta = tf.constant(l2_reg_beta)

      # Variables.
      # Input shape is batch_size x nfeatures
      # w1 is hidden layer: nfeatures x nhidden
      # b1 is: nhidden x 1
      # w2 is output (softmax) layer: nhidden x num_labels
      # b2 is: num_labels x 1
      w1 = tf.Variable(tf.truncated_normal([nfeatures, nhidden]))
      b1 = tf.Variable(tf.zeros([nhidden]))
      w2 = tf.Variable(tf.truncated_normal([nhidden, num_labels]))
      b2 = tf.Variable(tf.zeros([num_labels]))

      # Training computation. Adding dropout to hidden layer output.
      hidden_out = tf.nn.dropout(
        tf.nn.relu(tf.matmul(tf_train_dataset, w1) + b1),
        keep_prob=tf_dropout_keep_prob)
      logits = tf.matmul(hidden_out, w2) + b2
      loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
      reg_loss = loss + beta * tf.nn.l2_loss(w1) + beta * tf.nn.l2_loss(w2)

      # Optimizer.
      global_step = tf.Variable(0)  # count the number of steps taken.
      learning_rate = tf.train.exponential_decay(
        learning_rate=learning_rate,
        global_step=global_step,
        decay_steps=1000,
        decay_rate=0.96) 
      optimizer = tf.train.GradientDescentOptimizer(
            learning_rate).minimize(reg_loss, global_step=global_step)

      # Predictions for the training, validation, and test data.
      train_prediction = tf.nn.softmax(logits)
      valid_prediction = tf.nn.softmax(
              tf.matmul(
                  tf.nn.relu(tf.matmul(tf_valid_dataset, w1) + b1),
                  w2) + b2)
      test_prediction = tf.nn.softmax(
              tf.matmul(
                  tf.nn.relu(tf.matmul(tf_test_dataset, w1) + b1),
                  w2) + b2)

    with tf.Session(graph=graph) as session:

      # Enable this to emit logs for TensorBoard.
      #writer = tf.train.SummaryWriter("/tmp/tflogs", session.graph_def)

      tf.initialize_all_variables().run()
      print("Initialized")
      for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {
            tf_train_dataset: batch_data,
            tf_train_labels: batch_labels,
            tf_dropout_keep_prob: dropout_keep_prob}
        _, l, predictions = session.run(
          [optimizer, reg_loss, train_prediction], feed_dict=feed_dict)
        if (step % 100 == 0):
          print("Minibatch loss at step %d: %f" % (step, l))
          print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
          print("Validation accuracy: %.1f%%" % accuracy(
            valid_prediction.eval(), valid_labels))
      print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


if __name__ == '__main__':
    #run_sgd_with_reg()

    run_sgd_with_hidden_layer_with_reg(
        num_steps=3711,
        learning_rate=0.01,
        l2_reg_beta=0.005,
        dropout_keep_prob=0.8)
