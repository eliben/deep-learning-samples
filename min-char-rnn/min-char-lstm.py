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
data_size, vocab_size = len(data), len(chars)
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
hidden_size = 100
seq_length = 16 # number of steps to unroll the LSTM for
learning_rate = 1e-1

# Size of combined state (HV): input with hidden.
combined_size = hidden_size + vocab_size

# Stop when processed this much data
MAX_DATA = 1000000

# Model parameters/weights -- these are shared among all steps. Weights
# initialized randomly; biases initialized to 0.
# Inputs are characters one-hot encoded in a vocab-sized vector.
# Dimensions: H = hidden_size, V = vocab_size, HV = hidden_size + vocab_size
Wf = np.random.randn(hidden_size, combined_size) * 0.01
bf = np.zeros((hidden_size, 1))
Wi = np.random.randn(hidden_size, combined_size) * 0.01
bi = np.zeros((hidden_size, 1))
Wct = np.random.randn(hidden_size, combined_size) * 0.01
bct = np.zeros((hidden_size, 1))
Wo = np.random.randn(hidden_size, combined_size) * 0.01
bo = np.zeros((hidden_size, 1))
Why = np.random.randn(vocab_size, hidden_size) * 0.01
by = np.zeros((vocab_size, 1))
