# This is a generic linear classifier that implements SGD - Stochastic Gradient
# Descent (actually its mini-batch generalization).
#
# It has to be derived from by classes that provide a 'loss' member function,
# to implement different classifiers.
# See http://cs231n.github.io/classification/ for background.
import numpy as np
import random

class LinearClassifier:
    def __init__(self):
        self.W = None

    def train(self,
              X,
              y,
              learning_rate=1e-3,
              reg=1e-5,
              num_iters=100,
              batch_size=200,
              verbose=False):
        """Train this linear classifier using stochastic gradient descent.

        Inputs:
        - X: D x N array of training data. Each training point is a
             D-dimensional column.
        - y: 1-dimensional array of length N with labels 0...K-1, for K classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training
        iteration.
        """
        D, N = X.shape
        K = np.max(y) + 1
        if self.W is None:
            # Lazily initialize W to a random matrix
            self.W = np.random.randn(K, D) * 0.001

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in xrange(num_iters):
            batch_samples = np.random.choice(N, batch_size)
            X_batch = X[:, batch_samples]
            y_batch = y[batch_samples]

            # Evaluate loss and gradient
            loss, dW = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            self.W += -learning_rate * dW

            if verbose and it % 100 == 0:
                print 'iteration %d / %d: loss %f' % (it, num_iters, loss)
        return loss_history

    def predict(self, X):
        """Use the trained weights of this linear classifier to predict labels.

        Inputs:
        - X: D x N array of training data. Each column is a D-dimensional point.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        y_pred = self.W.dot(X)
        return y_pred.argmax(axis=0)
