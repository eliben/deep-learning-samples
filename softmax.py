# Softmax classifier loss function
import numpy as np
from random import shuffle

import linear_classifier


def softmax_loss_naive(W, X, y, reg):
    """Softmax loss function, naive implementation (with loops)

    Important dimensions: K is number of classes we classify samples to. D is
    the dimensionality of data (for example, 32x32x3 images have D=3072). Note
    that bias is often folded into the sample as "1", so the actual
    dimensionality may be +1 (or 3073 for those images).
    N is simply the number of samples we're working with.

    Inputs:
      - W: K x D array of weights.
      - X: D x N array of data. Each datum is a (D-dimensional) column.
      - y: 1-dimensional array of length N with labels 0...K-1, for K classes.
           y[i] is the correct classification of sample i.
      - reg: (float) regularization strength

    Returns a tuple of:
      - loss as single float
      - gradient with respect to weights W; an array of same shape as W
    """
    # Note: this code is from the internet, since I couldn't find an explanation
    # of how to compute the softmax gradient in lecture notes.

    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    for i in range(X.shape[1]):
        scores = W.dot(X[:, i])

        # Shift down by max to improve numerical stability -- now the highest
        # number is 0.
        scores -= np.max(scores)
        prob = 0.0
        loss -= scores[y[i]]

        for curr_score in scores:
            prob += np.exp(curr_score)

        for j in range(W.shape[0]):
            prob_ji = np.exp(scores[j]) / prob
            margin = -prob_ji * X[:, i].T

            if j == y[i]:
                margin = (1 - prob_ji) * X[:, i].T
            dW[j, :] += -margin

        loss += np.log(prob)

    loss /= X.shape[1]
    dW /= X.shape[1]

    # Regularization
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    return loss, dW


class Softmax(linear_classifier.LinearClassifier):
    """ A subclass that uses the Softmax + cross entropy loss function """

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_naive(self.W, X_batch, y_batch, reg)
