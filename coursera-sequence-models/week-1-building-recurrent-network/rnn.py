import numpy as np
from rnn_utils import sigmoid, softmax


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

    a_next = a0

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


def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    """
    Implement a single forward step of the LSTM-cell as described in Figure (4)

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    c_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
      Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
      bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
      Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
      bi -- Bias of the update gate, numpy array of shape (n_a, 1)
      Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
      bc --  Bias of the first "tanh", numpy array of shape (n_a, 1)
      Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
      bo --  Bias of the output gate, numpy array of shape (n_a, 1)
      Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
      by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    c_next -- next memory state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass,
             contains (a_next, c_next, a_prev, c_prev, xt, parameters)

    Note: ft/it/ot stand for the forget/update/output gates, cct stands for the
          candidate value (c tilde), c stands for the memory value
    """

    # Retrieve parameters from "parameters"
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wi = parameters["Wi"]
    bi = parameters["bi"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    Wy = parameters["Wy"]
    by = parameters["by"]

    # Retrieve dimensions from shapes of xt and Wy
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    # Concatenate a_prev and xt
    concat = np.zeros((n_a + n_x, m))
    concat[: n_a, :] = a_prev
    concat[n_a :, :] = xt

    # Compute values for ft, it, cct, c_next, ot, a_next using the formulas
    # given figure (4)
    ft = sigmoid(np.dot(Wf, concat) + bf)
    it = sigmoid(np.dot(Wi, concat) + bi)
    cct = np.tanh(np.dot(Wc, concat) + bc)
    c_next = ft * c_prev + it * cct
    ot = sigmoid(np.dot(Wo, concat) + bo)
    a_next = ot * np.tanh(c_next)

    yt_pred = softmax(np.dot(Wy, a_next) + by)

    # store values needed for backward propagation in cache
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)
    return a_next, c_next, yt_pred, cache


def lstm_forward(x, a0, parameters):
    """
    Implement the forward propagation of the recurrent neural network using an
    LSTM-cell described in Figure (3).

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
      Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
      bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
      Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
      bi -- Bias of the update gate, numpy array of shape (n_a, 1)
      Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
      bc -- Bias of the first "tanh", numpy array of shape (n_a, 1)
      Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
      bo -- Bias of the output gate, numpy array of shape (n_a, 1)
      Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
      by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of all the caches, x)
    """

    # Initialize "caches", which will track the list of all the caches
    caches = []

    # Retrieve dimensions from shapes of x and parameters['Wy'] (≈2 lines)
    n_x, m, T_x = x.shape
    n_y, n_a = parameters['Wy'].shape

    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))

    # Initialize a_next and c_next
    a_next = a0
    c_next = np.zeros_like(a_next)

    # loop over all time-steps
    for t in range(T_x):
        # Update next hidden state, next memory state, compute the prediction,
        # get the cache.
        a_next, c_next, yt, cache = lstm_cell_forward(x[:,:,t], a_next, c_next,
                                                      parameters)
        a[:,:,t] = a_next
        y[:,:,t] = yt
        c[:,:,t] = c_next
        caches.append(cache)

    # store values needed for backward propagation in cache
    caches = (caches, x)

    return a, y, c, caches


if __name__ == '__main__':
    np.random.seed(1)
    x = np.random.randn(3,10,7)
    a0 = np.random.randn(5,10)
    Wf = np.random.randn(5, 5+3)
    bf = np.random.randn(5,1)
    Wi = np.random.randn(5, 5+3)
    bi = np.random.randn(5,1)
    Wo = np.random.randn(5, 5+3)
    bo = np.random.randn(5,1)
    Wc = np.random.randn(5, 5+3)
    bc = np.random.randn(5,1)
    Wy = np.random.randn(2,5)
    by = np.random.randn(2,1)

    parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

    a, y, c, caches = lstm_forward(x, a0, parameters)
    print("a[4][3][6] = ", a[4][3][6])
    print("a.shape = ", a.shape)
    print("y[1][4][3] =", y[1][4][3])
    print("y.shape = ", y.shape)
    print("caches[1][1[1]] =", caches[1][1][1])
    print("c[1][2][1]", c[1][2][1])
    print("len(caches) = ", len(caches))


    #np.random.seed(1)
    #xt = np.random.randn(3,10)
    #a_prev = np.random.randn(5,10)
    #c_prev = np.random.randn(5,10)
    #Wf = np.random.randn(5, 5+3)
    #bf = np.random.randn(5,1)
    #Wi = np.random.randn(5, 5+3)
    #bi = np.random.randn(5,1)
    #Wo = np.random.randn(5, 5+3)
    #bo = np.random.randn(5,1)
    #Wc = np.random.randn(5, 5+3)
    #bc = np.random.randn(5,1)
    #Wy = np.random.randn(2,5)
    #by = np.random.randn(2,1)

    #parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

    #a_next, c_next, yt, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)
    #print("a_next[4] = ", a_next[4])
    #print("a_next.shape = ", c_next.shape)
    #print("c_next[2] = ", c_next[2])
    #print("c_next.shape = ", c_next.shape)
    #print("yt[1] =", yt[1])
    #print("yt.shape = ", yt.shape)
    #print("cache[1][3] =", cache[1][3])
    #print("len(cache) = ", len(cache))
