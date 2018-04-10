import numpy as np
from rnn_utils import softmax


def rnn_cell_forward(xt, a_prev, parameters):
    """
    Implements a single forward step of the RNN-cell as described in Figure (2)

    Vectorized over 'm' samples.

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
      Wax -- Weight matrix multiplying the input,
             numpy array of shape (n_a, n_x)
      Waa -- Weight matrix multiplying the hidden state, numpy array of
             shape (n_a, n_a)
      Wya -- Weight matrix relating the hidden-state to the output,
             numpy array of shape (n_y, n_a)
      ba -- Bias, numpy array of shape (n_a, 1)
      by -- Bias relating the hidden-state to the output, numpy array
            of shape (n_y, 1)
    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains
             (a_next, a_prev, xt, parameters)
    """
    # Retrieve parameters from "parameters"
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba)
    yt_pred = softmax(np.dot(Wya, a_next) + by)

    # store values you need for backward propagation in cache
    cache = (a_next, a_prev, xt, parameters)
    return a_next, yt_pred, cache


def rnn_forward(x, a0, parameters):
    """
    Implement the forward propagation of the recurrent neural network described
    in Figure (3).

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
      Waa -- Weight matrix multiplying the hidden state, numpy array
             of shape (n_a, n_a)
      Wax -- Weight matrix multiplying the input, numpy array
             of shape (n_a, n_x)
      Wya -- Weight matrix relating the hidden-state to the output, numpy array
             of shape (n_y, n_a)
      ba -- Bias numpy array of shape (n_a, 1)
      by -- Bias relating the hidden-state to the output, numpy array
            of shape (n_y, 1)

    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y_pred -- Predictions for every time-step, numpy array
              of shape (n_y, m, T_x)
    caches -- tuple of values needed for the backward pass,
              contains (list of caches, x)
    """

    # Initialize "caches" which will contain the list of all caches
    caches = []

    # Retrieve dimensions from shapes of x and parameters["Wya"]
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape

    # Hidden states have shape:
    #   n_a, m, T_x (hidden dimension, num samples, seq index)
    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))

    a_next = a[:,:,0]

    # loop over all time-steps
    for t in range(0, T_x):
        # Update next hidden state, compute the prediction, get the cache
        a_next, yt_pred, cache = rnn_cell_forward(x[:,:,t], a_next, parameters)
        a[:,:,t] = a_next
        y_pred[:,:,t] = yt_pred
        caches.append(cache)

    # store values needed for backward propagation in cache
    caches = (caches, x)
    return a, y_pred, caches


if __name__ == '__main__':
    #np.random.seed(1)
    #xt = np.random.randn(3,10)
    #a_prev = np.random.randn(5,10)
    #Waa = np.random.randn(5,5)
    #Wax = np.random.randn(5,3)
    #Wya = np.random.randn(2,5)
    #ba = np.random.randn(5,1)
    #by = np.random.randn(2,1)
    #parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}

    #a_next, yt_pred, cache = rnn_cell_forward(xt, a_prev, parameters)
    #print("a_next[4] = ", a_next[4])
    #print("a_next.shape = ", a_next.shape)
    #print("yt_pred[1] =", yt_pred[1])
    #print("yt_pred.shape = ", yt_pred.shape)

    np.random.seed(1)
    x = np.random.randn(3,10,4)
    a0 = np.random.randn(5,10)
    Waa = np.random.randn(5,5)
    Wax = np.random.randn(5,3)
    Wya = np.random.randn(2,5)
    ba = np.random.randn(5,1)
    by = np.random.randn(2,1)
    parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}

    a, y_pred, caches = rnn_forward(x, a0, parameters)
    print("a[4][1] = ", a[4][1])
    print("a.shape = ", a.shape)
    print("y_pred[1][3] =", y_pred[1][3])
    print("y_pred.shape = ", y_pred.shape)
    print("caches[1][1][3] =", caches[1][1][3])
    print("len(caches) = ", len(caches))
