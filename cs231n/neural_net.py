# Simple neutral net classifier
#
# See http://cs231n.github.io/neural-networks-case-study
import numpy as np


def two_layer_net(X, model, y=None, reg=0.0):
    """Run two-layer fully connected NN.

      The net has an input dimension of D, a hidden layer dimension of H, and
      performs classification over C classes. We use a softmax loss function and
      L2 regularization of the weight matrices. The two layer net should use a
      ReLU nonlinearity after the first affine layer.

      The two layer net has the following architecture (FC = fully connected
      layer):

      [input] -- [FC] -- [ReLU] -- [FC] -- [softmax]

      The outputs of the second fully-connected layer are the scores for each
      class.

      Inputs:
      - X: Input data of shape (N, D). Each X[i] is a training sample.
      - model: Dictionary mapping parameter names to arrays of parameter values.
        It should contain the following:
        - W1: First layer weights; has shape (D, H)
        - b1: First layer biases; has shape (H,)
        - W2: Second layer weights; has shape (H, C)
        - b2: Second layer biases; has shape (C,)
      - y: Vector of training labels. y[i] is the label for X[i], and each
        y[i] is an integer in the range 0 <= y[i] < C. This parameter is
        optional; if it is not passed then we only return scores, and if it is
        passed then we instead return the loss and gradients.
      - reg: Regularization strength.

      Returns:
      If y not is passed, return a matrix scores of shape (N, C)
      where scores[i, c] is the score for class c on input X[i].

      If y is passed, instead return a tuple of:
      - loss: Loss (data loss and regularization loss) for this batch of
        training samples.
      - grads: Dictionary mapping parameter names to gradients of those
        parameters with respect to the loss function. This should have the same
        keys as model.
    """
    # unpack variables from the model dictionary
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    N, D = X.shape

    # Compute the forward pass: layer 1, ReLU, layer 2
    Hout = X.dot(W1) + b1
    ReLU = np.maximum(0, Hout)
    scores = ReLU.dot(W2) + b2

    # The shape of scores is (N, C) -- for each input it has an array of scores
    # for each of the classification classes. scores[i][c] is the score of input
    # X[i] for class c.

    # The softmax data loss is defined as follows: for every input i, we have an
    # array F holding its scores for C classes (in the notation above, F is one
    # line of 'scores'). Li is the data loss for input i. The total data loss
    # for all inputs is the average:
    #
    # L = 1/N * Sum_i Li
    #
    # Each Li is:
    #
    # Li = -log( exp(F[y[i]]) / (Sum_j exp(F[j])))
    #
    # Where y[i] is the correct class for input i.

    # Compute the expression inside the log for all possible scores, and then
    # select only the relevant ones. probs's shape is (N, C) just like scores,
    # since it collects losses for all possible classes. correct_probs only
    # selects the losses for the correct classes, which is what we need. It
    # selects one column in each row, resulting in shape (N,)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    correct_probs = probs[range(N), y]

    # Finally compute the loss for all examples
    data_loss = np.sum(-np.log(correct_probs)) / N

    # Regularization loss is sum of 1/2 * reg * w^2 for every weight in the
    # model.
    reg_loss = 0.5 * reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2))

    # Compute the loss.
    loss = data_loss + reg_loss

    # Compute the gradients based on
    # http://cs231n.github.io/neural-networks-case-study/
    dscores = probs
    dscores[range(N), y] -= 1
    dscores /= N

    grads = {}
    grads['W2'] = Hout.T.dot(dscores) + reg * W2
    grads['b2'] = np.sum(dscores, axis=0)

    # Next backprop into hidden layer
    dhidden = dscores.dot(W2.T)
    # Backprop the ReLU non-linearity
    dhidden[Hout <= 0] = 0
    grads['W1'] = X.T.dot(dhidden) + reg * W1
    grads['b1'] = np.sum(dhidden, axis=0)

    # Return scores or (loss, grads) based on y.
    if y is None:
        return scores
    else:
        return loss, grads
