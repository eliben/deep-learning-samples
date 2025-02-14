# Based on the notebook
# https://github.com/udlbook/udlbook/blob/main/Notebooks/Chap12/12_1_Self_Attention.ipynb

import numpy as np
import matplotlib.pyplot as plt

# Seeds follow the order/values of the notebook (f)
np.random.seed(3)

# Number of inputs
N = 3
# Number of dimensions of each input
D = 4

# The self-attention mechanism maps N inputs x_n to N outputs x'_n.
# Inputs and outputs are (D,1) arrays.

all_x = []
for n in range(N):
    all_x.append(np.random.normal(size=(D, 1)))
print(all_x)

np.random.seed(0)

# Attention parameters for Q,K,V: matrices are (D,D) arrays, with corresponding
# biases that are (D,1) arrays.
omega_q = np.random.normal(size=(D, D))
omega_k = np.random.normal(size=(D, D))
omega_v = np.random.normal(size=(D, D))
beta_q = np.random.normal(size=(D, 1))
beta_k = np.random.normal(size=(D, 1))
beta_v = np.random.normal(size=(D, 1))

# Compute Q,K,V for each input
all_queries = []
all_keys = []
all_values = []

for x in all_x:
    # Shapes: (D, 1) = (D, D) @ (D, 1) + (D, 1)
    query = omega_q @ x + beta_q
    key = omega_k @ x + beta_k
    value = omega_v @ x + beta_v

    all_queries.append(query)
    all_keys.append(key)
    all_values.append(value)


def softmax(x):
    """Compute the softmax of vector (or list) x."""
    exps = np.exp(x)
    return exps / np.sum(exps)


# Output
all_x_prime = []

# For each output
for n in range(N):
    # A list of dot products of query n with all keys.
    # all_km_qn[i] is the dot product of key i with query n. It's a list with
    # N elements.
    all_km_qn = []
    for key in all_keys:
        dot_product = all_queries[n].T @ key
        all_km_qn.append(dot_product[0][0])

    attention = softmax(all_km_qn)
    print(f"Attentions for output n={n}: {attention}")

    # Compute self-attention: a weighted sum of all values.
    x_prime = np.zeros((D, 1))
    for i in range(len(all_values)):
        x_prime += attention[i] * all_values[i]

    all_x_prime.append(x_prime)

# Print out true values to check you have it correct
print("x_prime_0_calculated:", all_x_prime[0].transpose())
print("x_prime_0_true: [[ 0.94744244 -0.24348429 -0.91310441 -0.44522983]]")
print("x_prime_1_calculated:", all_x_prime[1].transpose())
print("x_prime_1_true: [[ 1.64201168 -0.08470004  4.02764044  2.18690791]]")
print("x_prime_2_calculated:", all_x_prime[2].transpose())
print("x_prime_2_true: [[ 1.61949281 -0.06641533  3.96863308  2.15858316]]")


# Vector formulation


def softmax_cols(data_in):
    # Exponentiate all of the values
    exp_values = np.exp(data_in)
    # Sum over columns
    denom = np.sum(exp_values, axis=0)
    # Replicate denominator to N rows
    denom = np.matmul(np.ones((data_in.shape[0], 1)), denom[np.newaxis, :])
    # Compute softmax
    softmax = exp_values / denom
    # return the answer
    return softmax


# X: (D,N) matrix of inputs
# omega_v, omega_q, omega_k: (D,D) matrices
# beta_v, beta_q, beta_k: (D,1) vectors
def self_attention(
    X, omega_v, omega_q, omega_k, beta_v, beta_q, beta_k, is_scaled=False
):

    Q = omega_q @ X + beta_q
    K = omega_k @ X + beta_k
    V = omega_v @ X + beta_v

    KQ = K.T @ Q
    if is_scaled:
        KQ /= np.sqrt(K.shape[0])
    att = softmax_cols(KQ)
    print("attention matrix:", att)
    X_prime = V @ att

    return X_prime


# Represent all inputs as a (D,N) matrix. Each column is an input.
X = np.column_stack(all_x)
print(self_attention(X, omega_v, omega_q, omega_k, beta_v, beta_q, beta_k))

print(
    self_attention(X, omega_v, omega_q, omega_k, beta_v, beta_q, beta_k, is_scaled=True)
)
