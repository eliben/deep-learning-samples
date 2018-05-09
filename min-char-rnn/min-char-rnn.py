"""
Taken from Andrej Karpathy's min-char-rnn:
    https://gist.github.com/karpathy/d4dee566867f8291f086

Modified in various ways for better introspection / customization, and added
comments.

----

Minimal character-level Vanilla RNN model.
Written by Andrej Karpathy (@karpathy)
BSD License
"""
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

# Numerical index [0:N) mapping for N chars, to be used as vector indices (for
# one-hot vectors, etc).
char_to_ix = {ch:i for i, ch in enumerate(chars)}
ix_to_char = {i:ch for i, ch in enumerate(chars)}
print('char_to_ix', char_to_ix)
print('ix_to_char', ix_to_char)

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 16 # number of steps to unroll the RNN for
learning_rate = 1e-1

# Stop when processed this much data
MAX_DATA = 1000000

# Model parameters/weights -- these are shared among all steps. Weights
# initialized randomly; biases initialized to 0.
# Inputs are characters one-hot encoded in a vocab-sized vector.
# Dimensions: H = hidden_size, V = vocab_size
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01       # input to hidden
Whh = np.random.randn(hidden_size, hidden_size) * 0.01      # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size) * 0.01       # hidden to output
bh = np.zeros((hidden_size, 1))     # hidden bias
by = np.zeros((vocab_size, 1))      # output bias

def lossFun(inputs, targets, hprev):
  """Runs forward and backward passes through the RNN.

  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  # Caches that keep values computed in the forward pass at each time step, to
  # be reused in the backward pass.
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  # Forward pass
  for t in xrange(len(inputs)):
    # Input at time step t is xs[t] -- a one-hot encoded vector of shape (V, 1).
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1

    # Compute h[t] from h[t-1] and x[t]
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)

    # Compute ps[t] - probabilities for output, and the loss using xentropy.
    ys[t] = np.dot(Why, hs[t]) + by
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
    loss += -np.log(ps[t][targets[t],0])

  # Backward pass: compute gradients going backwards.
  # Gradients are initialized to 0s, and every time step contributes to them.
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])

  for t in reversed(xrange(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here

    # Applying chain rule to propagate grad into dWhy and dby.
    #
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)

  # Gradient clipping
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam)

  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
  """
  sample a sequence of integers from the model
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in xrange(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0

# gradient checking
from random import uniform
def gradCheck(inputs, targets, hprev):
  global Wxh, Whh, Why, bh, by
  num_checks, delta = 10, 1e-5
  _, dWxh, dWhh, dWhy, dbh, dby, _ = lossFun(inputs, targets, hprev)
  for param,dparam,name in zip([Wxh, Whh, Why, bh, by],
                               [dWxh, dWhh, dWhy, dbh, dby],
                               ['Wxh', 'Whh', 'Why', 'bh', 'by']):
    s0 = dparam.shape
    s1 = param.shape
    assert s0 == s1, 'Error dims dont match: %s and %s.' % (`s0`, `s1`)
    print(name)
    for i in xrange(num_checks):
      ri = int(uniform(0,param.size))
      # evaluate cost at [x + delta] and [x - delta]
      old_val = param.flat[ri]
      param.flat[ri] = old_val + delta
      cg0, _, _, _, _, _, _ = lossFun(inputs, targets, hprev)
      param.flat[ri] = old_val - delta
      cg1, _, _, _, _, _, _ = lossFun(inputs, targets, hprev)
      param.flat[ri] = old_val # reset old value for this parameter
      # fetch both numerical and analytic gradient
      grad_analytic = dparam.flat[ri]
      grad_numerical = (cg0 - cg1) / ( 2 * delta )
      rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
      print('%f, %f => %e ' % (grad_numerical, grad_analytic, rel_error))
      # rel_error should be on order of 1e-7 or less

def basicGradCheck():
  inputs = [char_to_ix[ch] for ch in data[:seq_length]]
  targets = [char_to_ix[ch] for ch in data[1:seq_length+1]]
  hprev = np.zeros((hidden_size,1)) # reset RNN memory
  gradCheck(inputs, targets, hprev)

basicGradCheck()

while p < MAX_DATA:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0:
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  # sample from the model now and then
  if n % 1000 == 0:
    sample_ix = sample(hprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print('----\n %s \n----' % (txt, ))

  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 200 == 0: print('iter %d (p=%d), loss: %f' % (n, p, smooth_loss))

  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                [dWxh, dWhh, dWhy, dbh, dby],
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += seq_length # move data pointer
  n += 1 # iteration counter
