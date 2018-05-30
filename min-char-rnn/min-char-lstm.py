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



