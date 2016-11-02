# Common library code for running logistic regressions and classifiers.
#
# Eli Bendersky (http://eli.thegreenplace.net)
# This code is in the public domain
import numpy as np


def augment_1s_column(X):
    """Augments the given data matrix with a first column of ones.

    X: (k, n) k rows of data items, each having n features.

    Returns a (k, n+1) matrix with an additional column of 1s at the start.
    """
    return np.hstack((np.ones((X.shape[0], 1)), X))


def feature_normalize(X):
    """Normalizes the feature matrix X.

    Given a feature matrix X, where each row is a vector of features, normalizes
    each feature. Returns (X_norm, mu, sigma) where mu and sigma are the mean
    and stddev of features (vectors).

    Where stddev is zero for a feature, it's clamped to one. In X that means
    all items had the same value for the feature. For normalizing other data,
    sigma=1 means the feature remains its value with mean subtracted, and no
    scaling.
    """
    num_features = X.shape[1]
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


def predict_binary(X, theta):
    """Makes classification predictions for the data in X using theta.

    For a given data item x, the prediction is +1 if x.dot(theta) >= 0, and -1
    otherwise. Note that this also works for logistic regression since for
    x.dot(theta) >= 0 the sigmoid is >= 0.5 which we also consider +1.

    X: (k, n) k rows of data items, each having n features; augmented.
    theta: (n, 1) regression parameters.

    Returns yhat (k, 1) - either +1 or -1 classification for each item.
    """
    yhat = X.dot(theta)
    # Fix the cases where yhat == 0 to be positive; otherwise np.sign would
    # return 0. Note that it should be exceedingly rare in practice to get an
    # exact 0 for some result.
    yhat[yhat == 0] = 1
    return np.sign(yhat)


def sigmoid(z):
    """Computes sigmoid function.

    z: array of input values.

    Returns array of outputs, sigmoid(z).
    """
    # Note: this version of sigmoid tries to avoid overflows in the computation
    # of e^(-z), by using an alternative formulation when z is negative, to get
    # 0. e^z / (1+e^z) is equivalent to the definition of sigmoid, but we won't
    # get e^(-z) to overflow when z is very negative.
    # Since both the x and y arguments to np.where are evaluated by Python, we
    # may still get overflow warnings for large z elements; therefore we ignore
    # warnings during this computation.
    with np.errstate(over='ignore', invalid='ignore'):
        return np.where(z >= 0,
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z)))


def softmax_layer(X, W):
    """Computes the softmax layer for inputs X and weights W.

    Works in a vectorized manner for a whole batch of k data items.

    X: (k, n) k rows of data items, each having n features; augmented.
    W: (n, t) t is the number of classes.

    Applies the softmax function to X.dot(W). Returns an array of probabilities
    with shape (k, t). Each row is the probabilities of the t classes for that
    datum.
    """
    # Compute logits: (k, t)
    logits = X.dot(W)
    # Vectorized sofmax computation on every row of logits. The normalization
    # is done as a sum of columns, which is then broadcast over exp_logits.
    # The result is also (k, t)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def softmax_cross_entropy_loss(X, y, W, reg_beta=0.0):
    """Softmax cross entropy loss and gradient computation.

    X: (k, n) k rows of data items, each having n features; augmented.
    y: (k,) a number in the closed interval [0..t-1] - the correct class, for
            each of the k inputs.
    W: (n, t) t is the number of classes.

    Applies the xent loss (comparing with y) for softmax_layer and computes its
    gradient w.r.t. W. Returns (loss, dW).
    """
    softmax = softmax_layer(X, W)

    # y contains class IDs -- numbers in range [0, t) -- which is useful for the
    # selection we need for xent.
    # For each sample, we have to select the "right" softmax result and log it.
    # The right one comes at index for any given input. Since in softmax the
    # inputs are rows, [range(k), y] selects the yth in each row in a vectorized
    # manner.
    k = X.shape[0]
    logprobs = -np.log(softmax[range(k), y])
    loss = np.sum(logprobs) / k

    # Add regularization loss.
    loss += np.sum(W * W) * reg_beta / 2

    # Compute gradient for the weights in a vectorized way. Again for vectorized
    # indexing [range(k), y] selects the yth weight in each row.
    dW = softmax
    dW[range(k), y] -= 1
    dW = np.dot(X.T, dW) / k

    # Gradient of regularization.
    dW += W * reg_beta
    return loss, dW


def predict_logistic_probability(X, theta):
    """Makes classification predictions for the data in X using theta.

    X: (k, n) k rows of data items, each having n features; augmented.
    theta: (n, 1) logistic regression parameters.

    Computes the logistic regression prediction. Returns yhat (k, 1) - number
    in the range (0.0, 1.0) for each item. The number is the probability that
    the item is classified as +1.
    """
    z = X.dot(theta)
    return sigmoid(z)


def cross_entropy_loss_binary(X, y, theta, reg_beta=0.0):
    """Computes the cross-entropy loss for binary classification.

    Applies the cross-entropy loss function to sigmoid activation for X and
    theta.

    See square_loss for arguments and return value.
    """
    k, n = X.shape
    # Calls to predict_logistic_probability may produce probabilities that are
    # 0.0 and 1.0, due to the exponent in sigmoid and the large numbers
    # involved. np.log on 0.0 overflows, so we clip the input of np.log to
    # values very close to 0 instead.
    eps = np.finfo(np.float32).eps
    yhat_prob = np.clip(predict_logistic_probability(X, theta),
                        a_min=eps,
                        a_max=1.0-eps)
    loss = np.mean(np.where(y == 1,
                            -np.log(yhat_prob),
                            -np.log(1 - yhat_prob)))
    # Add regularization.
    loss += np.dot(theta.T, theta) * reg_beta / 2

    yh = np.where(y == 1, yhat_prob - 1, yhat_prob)
    dtheta = np.dot(yh.T, X).T / k + reg_beta * theta
    return loss.flat[0], dtheta


def square_loss(X, y, theta, reg_beta=0.0):
    """Computes squared loss and gradient.

    Based on mean square margin loss.

    X: (k, n) data items.
    y: (k, 1) result (+1 or -1) for each data item in X.
    theta: (n, 1) parameters.
    reg_beta: optional regularization strength, for L2 regularization.

    Returns (loss, dtheta) where loss is the aggregate numeric loss for this
    theta, and dtheta is (n, 1) gradients for theta based on that loss.

    Note: the mean (division by k) helps; otherwise, the loss is very large and
    a tiny learning rate is required to prevent divergence in the beginning of
    the search.
    """
    k, n = X.shape
    margin = y * X.dot(theta)
    diff = margin - 1
    loss = np.dot(diff.T, diff) / k + np.dot(theta.T, theta) * reg_beta / 2

    dtheta = np.zeros_like(theta)
    for j in range(n):
        dtheta[j, 0] = (2 * np.dot((diff * y).T, X[:, j]) / k +
                        reg_beta * theta[j, 0])
    return loss.flat[0], dtheta


def hinge_loss(X, y, theta, reg_beta=0.0):
    """Computes hinge loss and gradient.

    See square_loss for arguments and return value.
    """
    k, n = X.shape
    # margin is (k, 1)
    margin = y * X.dot(theta)
    loss = (np.sum(np.maximum(np.zeros_like(margin), 1 - margin)) / k +
            np.dot(theta.T, theta) * reg_beta / 2)

    dtheta = np.zeros_like(theta)
    # yx is (k, n) where the elementwise multiplication by y is broadcast across
    # the whole X.
    yx = y * X
    # We're going to select columns of yx, and each column turns into a vector.
    # Precompute the margin_selector vector which has for each j whether the
    # margin for that j was < 1.
    # Note: still keeping an explicit loop over n since I don't expect the
    # number of features to be very large. It's possible to fully vectorize this
    # but that would make the computation even more obscure. I'll do that if
    # performance becomes an issue with this version.
    margin_selector = (margin < 1).ravel()
    for j in range(n):
        # Sum up the contributions to the jth theta element gradient from all
        # input samples.
        dtheta[j, 0] = (np.sum(np.where(margin_selector, -yx[:, j], 0)) / k +
                        reg_beta * theta[j, 0])
    return loss.flat[0], dtheta


def generate_batch(X, y, batch_size=256):
    """Generate a randomized batch from X, y.

    X (k, n), y (k, 1): as usual.
    batch_size: size of the batch to create.

    Returns X_batch (batch_size, n), y_batch (batch_size, 1) pair.
    """
    batch_indices = np.random.choice(X.shape[0], batch_size, replace=False)
    return X[batch_indices, :], y[batch_indices]


def gradient_descent(X, y, init_theta,
                     lossfunc=None,
                     nsteps=100,
                     batch_size=None,
                     learning_rate=0.1):
    """Runs gradient descent optimization to minimize loss for X, y.

    Starts with init_theta. The shapes of X, y, init_theta have to work with
    the loss function provided. The actual theta array learned will have the
    same shape as init_theta.

    lossfunc:
        a function computing loss and gradients. Takes (X, y, theta).
        Returns (loss, dtheta) where loss is the numeric loss for this theta,
        and dtheta is the gradients for theta based on that loss (same shape
        as theta - every element is a gradient).
    nsteps: how many steps to run.
    batch_size:
        if None, the whole data set is used for every step. Otherwise
        batches of batch_size randomly-selected data items are used.
    learning_rate: learning rate update (multiplier of gradient).

    Yields 'nsteps + 1' pairs of (theta, loss). The first pair yielded is the
    initial theta and its loss; the rest carry results after each of the
    iteration steps.
    """
    theta = init_theta.copy()

    if batch_size is None:
        loss, dtheta = lossfunc(X, y, theta)
    else:
        X_batch, y_batch = generate_batch(X, y, batch_size)
        loss, dtheta = lossfunc(X_batch, y_batch, theta)
    yield theta, loss

    for step in range(nsteps):
        theta -= learning_rate * dtheta

        if batch_size is None:
            loss, dtheta = lossfunc(X, y, theta)
        else:
            X_batch, y_batch = generate_batch(X, y, batch_size)
            loss, dtheta = lossfunc(X_batch, y_batch, theta)
        yield theta, loss
