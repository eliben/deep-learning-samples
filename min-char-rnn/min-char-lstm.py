# Work in progress
from __future__ import print_function

import numpy as np
import sys

# Make it possible to provide input file as a command-line argument; input.txt
# is still the default.
if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = 'input.txt'

with open(filename, 'r') as f:
    data = f.read()

# All unique characters / entities in the data set.
chars = list(set(data))
data_size = len(data)
V = vocab_size = len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))

# Each character in the vocabulary gets a unique integer index assigned, in the
# half-open interval [0:N). These indices are useful to create one-hot encoded
# vectors that represent characters in numerical computations.
char_to_ix = {ch:i for i, ch in enumerate(chars)}
ix_to_char = {i:ch for i, ch in enumerate(chars)}
print('char_to_ix', char_to_ix)
print('ix_to_char', ix_to_char)

# Hyperparameters.

# Size of hidden state vectors; applies to h and c.
H = hidden_size = 100
seq_length = 16 # number of steps to unroll the LSTM for
learning_rate = 1e-1

# Size of combined state: input with hidden.
HV = H + V

# Stop when processed this much data
MAX_DATA = 1000000

# Model parameters/weights -- these are shared among all steps. Weights
# initialized randomly; biases initialized to 0.
# Inputs are characters one-hot encoded in a vocab-sized vector.
# Dimensions: H = hidden_size, V = vocab_size, HV = hidden_size + vocab_size
Wf = np.random.randn(H, HV) * 0.01
bf = np.zeros((H, 1))
Wi = np.random.randn(H, HV) * 0.01
bi = np.zeros((H, 1))
Wct = np.random.randn(H, HV) * 0.01
bct = np.zeros((H, 1))
Wo = np.random.randn(H, HV) * 0.01
bo = np.zeros((H, 1))
Why = np.random.randn(V, H) * 0.01
by = np.zeros((V, 1))


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


def lossFun(inputs, targets, hprev, cprev):
    """Runs forward and backward passes through the RNN.
 
      TODO: keep me updated!
      inputs, targets: Lists of integers. For some i, inputs[i] is the input
                       character (encoded as an index into the ix_to_char map)
                       and targets[i] is the corresponding next character in the
                       training data (similarly encoded).
      hprev: Hx1 array of initial hidden state
      cprev: Hx1 array of initial hidden state
 
      returns: ? loss, gradients on model parameters, and last hidden state
    """
    # Caches that keep values computed in the forward pass at each time step, to
    # be reused in the backward pass.
    xs, ys, hs, cs, fgs, igs, cts, ogs = {}, {}, {}, {}, {}, {}, {}, {}

    # Initial incoming states.
    hs[-1] = np.copy(hprev)
    cs[-1] = np.copy(cprev)

    loss = 0
    # Forward pass
    for t in range(len(inputs)):
        # Input at time step t is xs[t]. Prepare a one-hot encoded vector of
        # shape (V, 1). inputs[t] is the index where the 1 goes.
        xs[t] = np.zeros((V, 1))
        xs[t][inputs[t]] = 1

        # hprev and xs[t] are column vector; stack them together into a "taller"
        # column vector.
        xh = np.row_stack(hprev, xs[t])

        # Gates f, i and o.
        fgs[t] = sigmoid(np.dot(Wf, xh) + bf)
        igs[t] = sigmoid(np.dot(Wi, xh) + bi)
        ogs[t] = sigmoid(np.dot(Wo, xh) + bo)

        # Compute update candidate ct.
        cts[t] = np.tanh(np.dot(Wct, xh) + bct)

        # This step's h and c.
        cs[t] = fgs[t] * cs[t-1] + cts[t] * igs[t]
        hs[t] = np.tanh(cs[t]) * ogs[t]

        # Softmax for output.
        ys[t] = np.dot(Why, hs[t]) + by
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))

        # Cross-entropy loss.
        loss += -np.log(ps[t][targets[t], 0])

    # Initialize gradients of all weights/biases to 0.
    dWf = np.zeros_like(Wf)
    dbf = np.zeros_like(bf)
    dWi = np.zeros_like(Wi)
    dbi = np.zeros_like(bi)
    dWct = np.zeros_like(Wct)
    dbct = np.zeros_like(bct)
    dWo = np.zeros_like(Wo)
    dbo = np.zeros_like(bo)
    dWhy = np.zeros_like(Why)
    dby = np.zeros_like(by)

    # Incoming gradients for h and c
    dhnext = np.zeros_like(hs[0])
    dcnext = np.zeros_like(cs[0])

    # The backwards pass iterates over the input sequence backwards.
    for t in reversed(range(len(inputs))):
        # Backprop through the gradients of loss and softmax.
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1

        # Compute gradients for the Why and by parameters.
        dWhy += np.dot(dy, hs[t].T)
        dby += dy

        # Backprop through the fully-connected layer (Why, by) to h. Also add up the
        # incoming gradient for h from the next cell.
        dh = np.dot(Why.T, dy) + dhnext

        # Backprop through output gate and tanh to get to dc; dc also gets
        # dcnext added because it branches in two directions.
        dc = ogs[t] * (1 - hs[t] * hs[t]) * dh + dcnext
