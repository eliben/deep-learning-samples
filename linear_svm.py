# Linear SVM classifier.
# See http://cs231n.github.io/classification/ for background.
# And http://cs231n.github.io/optimization-1/ for the gradient parts.
import numpy as np
import random


def svm_loss_naive(W, X, y, reg):
    """Structured SVM loss function, naive implementation (with loops).

    Important dimensions: K is number of classes we classify samples to. D is
    the dimensionality of data (for example, 32x32x3 images have D=3072). Note
    that bias is often folded into the sample as "1", so the actual
    dimensionality may be +1 (or 3073 for those images).
    N is simply the number of samples we're working with.

    This function uses a delta value of 1.

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
    delta = 1
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    K = W.shape[0]
    N = X.shape[1]
    loss = 0.0
    for i in xrange(N):
        # Compute the loss for this sample.
        # The equation is:
        #
        #   Li = Sum_{j!=yi} max(0, wj*xi - wyi*xi + delta)
        #
        # We use W * Xi to find both wj*xi and wyi*xi, so we just index into
        # the result to find these distinct parts.
        #
        # X[:, i] is the ith column of X. scores now has the shape K x 1
        scores = W.dot(X[:, i])

        # wyi*xi is not changing in the sigma (internal loop), so precompute it.
        correct_class_score = scores[y[i]]

        # This computes the sigma.
        for j in xrange(K):
            margin = scores[j] - correct_class_score + delta
            if j == y[i]:
                continue
            if margin > 0:
                loss += margin

                # The gradient is only updated when margin > 0.
                dW[j, :] += X[:, i]
                dW[y[i], :] -= X[:, i]

    # Average the loss over N samples and add regularization.
    loss = (loss / N) + 0.5 * reg * np.sum(W * W)
    # Same for gradient.
    dW = (dW / N) + reg * W
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """Structured SVM loss function, vectorized implementation.

       Inputs and outputs are the same as svm_loss_naive.
    """
    delta = 1
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # scores's shape is (K, N): contains scores for all N samples, in columns.
    scores = W.dot(X)

    # We want to select the score of the correct class for every sample. Samples
    # are in columns. y[i] gives, for sample i, the correct class. Therefore
    # we need to index into every column at the appropriate y[i].
    correct_class_scores = scores[y, np.arange(scores.shape[1])]

    # Vectorized sum for all samples.
    s = (scores - correct_class_scores + delta).clip(min=0).sum(axis=0)

    # The sum was supposed to ignore the category with the correct score. But
    # for j=yi, the summed element is just max(0, delta), so we subtract delta
    # from the sums.
    s -= delta
    loss = np.mean(s) + 0.5 * reg * np.sum(W * W)

    b = (scores - correct_class_scores + delta) > 0

    print scores.shape, correct_class_scores.shape
    print s.shape
    print 'b', b.shape, b
    #print('y=', y)
    #print scores
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
