# Minimal character-based language model learning with an LSTM architecture.
#
# Overall code structure based on Andrej Karpathy's min-char-rnn model:
#    https://gist.github.com/karpathy/d4dee566867f8291f086
#
# But the architecture is modified to be LSTM rather than vanilla RNN.
# The companion blog post is:
#   https://eli.thegreenplace.net/2018/minimal-character-based-lstm-implementation/
#
# Tested with Python 3.6
#
# Eli Bendersky [http://eli.thegreenplace.net]
# BSD License per original (@karpathy)
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
learning_rate = 0.1

# The input x is concatenated with state h, and the joined vector is used to
# feed into most blocks within the LSTM cell. The combined height of the column
# vector is HV.
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
Wcc = np.random.randn(H, HV) * 0.01
bcc = np.zeros((H, 1))
Wo = np.random.randn(H, HV) * 0.01
bo = np.zeros((H, 1))
Wy = np.random.randn(V, H) * 0.01
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

      returns: loss, gradients on model parameters, and last hidden states
    """
    # Caches that keep values computed in the forward pass at each time step, to
    # be reused in the backward pass.
    xs, xhs, ys, ps, hs, cs, fgs, igs, ccs, ogs = (
            {}, {}, {}, {}, {}, {}, {}, {}, {}, {})

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
        # column vector - first the elements of x, then h.
        xhs[t] = np.vstack((xs[t], hs[t-1]))

        # Gates f, i and o.
        fgs[t] = sigmoid(np.dot(Wf, xhs[t]) + bf)
        igs[t] = sigmoid(np.dot(Wi, xhs[t]) + bi)
        ogs[t] = sigmoid(np.dot(Wo, xhs[t]) + bo)

        # Candidate cc.
        ccs[t] = np.tanh(np.dot(Wcc, xhs[t]) + bcc)

        # This step's h and c.
        cs[t] = fgs[t] * cs[t-1] + igs[t] * ccs[t]
        hs[t] = np.tanh(cs[t]) * ogs[t]

        # Softmax for output.
        ys[t] = np.dot(Wy, hs[t]) + by
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))

        # Cross-entropy loss.
        loss += -np.log(ps[t][targets[t], 0])

    # Initialize gradients of all weights/biases to 0.
    dWf = np.zeros_like(Wf)
    dbf = np.zeros_like(bf)
    dWi = np.zeros_like(Wi)
    dbi = np.zeros_like(bi)
    dWcc = np.zeros_like(Wcc)
    dbcc = np.zeros_like(bcc)
    dWo = np.zeros_like(Wo)
    dbo = np.zeros_like(bo)
    dWy = np.zeros_like(Wy)
    dby = np.zeros_like(by)

    # Incoming gradients for h and c; for backwards loop step these represent
    # dh[t] and dc[t]; we do truncated BPTT, so assume they are 0 initially.
    dhnext = np.zeros_like(hs[0])
    dcnext = np.zeros_like(cs[0])

    # The backwards pass iterates over the input sequence backwards.
    for t in reversed(range(len(inputs))):
        # Backprop through the gradients of loss and softmax.
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1

        # Compute gradients for the Wy and by parameters.
        dWy += np.dot(dy, hs[t].T)
        dby += dy

        # Backprop through the fully-connected layer (Wy, by) to h. Also add up
        # the incoming gradient for h from the next cell.
        dh = np.dot(Wy.T, dy) + dhnext

        # Backprop through multiplication with output gate; here "dtanh" means
        # the gradient at the output of tanh.
        dctanh = ogs[t] * dh
        # Backprop through the tanh function; since cs[t] branches in two
        # directions we add dcnext too.
        dc = dctanh * (1 - np.tanh(cs[t]) ** 2) + dcnext

        # Backprop through multiplication with the tanh; here "dhogs" means
        # the gradient at the output of the sigmoid of the output gate. Then
        # backprop through the sigmoid itself (ogs[t] is the sigmoid output).
        dhogs = dh * np.tanh(cs[t])
        dho = dhogs * ogs[t] * (1 - ogs[t])

        # Compute gradients for the output gate parameters.
        dWo += np.dot(dho, xhs[t].T)
        dbo += dho

        # Backprop dho to the xh input.
        dxh_from_o = np.dot(Wo.T, dho)

        # Backprop through the forget gate: sigmoid and elementwise mul.
        dhf = cs[t-1] * dc * fgs[t] * (1 - fgs[t])
        dWf += np.dot(dhf, xhs[t].T)
        dbf += dhf
        dxh_from_f = np.dot(Wf.T, dhf)

        # Backprop through the input gate: sigmoid and elementwise mul.
        dhi = ccs[t] * dc * igs[t] * (1 - igs[t])
        dWi += np.dot(dhi, xhs[t].T)
        dbi += dhi
        dxh_from_i = np.dot(Wi.T, dhi)

        dhcc = igs[t] * dc * (1 - ccs[t] ** 2)
        dWcc += np.dot(dhcc, xhs[t].T)
        dbcc += dhcc
        dxh_from_cc = np.dot(Wcc.T, dhcc)

        # Combine all contributions to dxh, and extract the gradient for the
        # h part to propagate backwards as dhnext.
        dxh = dxh_from_o + dxh_from_f + dxh_from_i + dxh_from_cc
        dhnext = dxh[V:, :]

        # dcnext from dc and the forget gate.
        dcnext = fgs[t] * dc

    # Gradient clipping to the range [-5, 5].
    for dparam in [dWf, dbf, dWi, dbi, dWcc, dbcc, dWo, dbo, dWy, dby]:
        np.clip(dparam, -5, 5, out=dparam)

    return (loss, dWf, dbf, dWi, dbi, dWcc, dbcc, dWo, dbo, dWy, dby,
            hs[len(inputs)-1], cs[len(inputs)-1])


def sample(h, c, seed_ix, n):
    """Sample a sequence of integers from the model.

    Runs the LSTM in forward mode for n steps; seed_ix is the seed letter for
    the first time step, h and c are the memory state. Returns a sequence of
    letters produced by the model (indices).
    """
    x = np.zeros((V, 1))
    x[seed_ix] = 1
    ixes = []

    for t in range(n):
        # Run the forward pass only.
        xh = np.vstack((x, h))
        fg = sigmoid(np.dot(Wf, xh) + bf)
        ig = sigmoid(np.dot(Wi, xh) + bi)
        og = sigmoid(np.dot(Wo, xh) + bo)
        cc = np.tanh(np.dot(Wcc, xh) + bcc)
        c = fg * c + ig * cc
        h = np.tanh(c) * og
        y = np.dot(Wy, h) + by
        p = np.exp(y) / np.sum(np.exp(y))

        # Sample from the distribution produced by softmax.
        ix = np.random.choice(range(V), p=p.ravel())
        x = np.zeros((V, 1))
        x[ix] = 1
        ixes.append(ix)
    return ixes


def gradCheck(inputs, targets, hprev, cprev):
    global Wf, Wi, bf, bi, Wcc, bcc, Wo, bo, Wy, by
    num_checks, delta = 10, 1e-5
    (_, dWf, dbf, dWi, dbi, dWcc, dbcc, dWo, dbo, dWy, dby,
     _, _) = lossFun(inputs, targets, hprev, cprev)
    for param, dparam, name in zip(
            [Wf, bf, Wi, bi, Wcc, bcc, Wo, bo, Wy, by],
            [dWf, dbf, dWi, dbi, dWcc, dbcc, dWo, dbo, dWy, dby],
            ['Wf', 'bf', 'Wi', 'bi', 'Wcc', 'bcc', 'Wo', 'bo', 'Wy', 'by']):
        assert dparam.shape == param.shape
        print(name)
        for i in range(num_checks):
            ri = np.random.randint(0, param.size)
            old_val = param.flat[ri]
            param.flat[ri] = old_val + delta
            numloss0 = lossFun(inputs, targets, hprev, cprev)[0]
            param.flat[ri] = old_val - delta
            numloss1 = lossFun(inputs, targets, hprev, cprev)[0]
            param.flat[ri] = old_val # reset
            grad_analytic = dparam.flat[ri]
            grad_numerical = (numloss0 - numloss1) / (2 * delta)
            if grad_numerical + grad_analytic == 0:
                rel_error = 0
            else:
                rel_error = (abs(grad_analytic - grad_numerical) /
                             abs(grad_numerical + grad_analytic))
            print('%s, %s => %e' % (grad_numerical, grad_analytic, rel_error))


def basicGradCheck():
    inputs = [char_to_ix[ch] for ch in data[:seq_length]]
    targets = [char_to_ix[ch] for ch in data[1:seq_length+1]]
    hprev = np.random.randn(H, 1)
    cprev = np.random.randn(H, 1)
    gradCheck(inputs, targets, hprev, cprev)

# Uncomment this to run gradient checking instead of training
#basicGradCheck()
#sys.exit()

# n is the iteration counter; p is the input sequence pointer, at the beginning
# of each step it points at the sequence in the input that will be used for
# training this iteration.
n, p = 0, 0

# Memory variables for Adagrad.
mWf = np.zeros_like(Wf)
mbf = np.zeros_like(bf)
mWi = np.zeros_like(Wi)
mbi = np.zeros_like(bi)
mWcc = np.zeros_like(Wcc)
mbcc = np.zeros_like(bcc)
mWo = np.zeros_like(Wo)
mbo = np.zeros_like(bo)
mWy = np.zeros_like(Wy)
mby = np.zeros_like(by)
smooth_loss = -np.log(1.0/V) * seq_length

while p < MAX_DATA:
    # Prepare inputs (we're sweeping from left to right in steps seq_length long)
    if p+seq_length+1 >= len(data) or n == 0:
        # Reset RNN memory
        hprev = np.zeros((H, 1))
        cprev = np.zeros((H, 1))
        p = 0 # go from start of data

    # In each step we unroll the RNN for seq_length cells, and present it with
    # seq_length inputs and seq_length target outputs to learn.
    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    # Sample from the model now and then.
    if n % 1000 == 0:
        sample_ix = sample(hprev, cprev, inputs[0], 200)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print('----\n %s \n----' % (txt,))

    # Forward seq_length characters through the RNN and fetch gradient.
    (loss, dWf, dbf, dWi, dbi, dWcc, dbcc, dWo, dbo, dWy, dby,
     hprev, cprev) = lossFun(inputs, targets, hprev, cprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if n % 200 == 0:
        print('iter %d (p=%d), loss %f' % (n, p, smooth_loss))

    # Perform parameter update with Adagrad.
    for param, dparam, mem in zip(
            [Wf, bf, Wi, bi, Wcc, bcc, Wo, bo, Wy, by],
            [dWf, dbf, dWi, dbi, dWcc, dbcc, dWo, dbo, dWy, dby],
            [mWf, mbf, mWi, mbi, mWcc, mbcc, mWo, mbo, mWy, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)

    p += seq_length
    n += 1
