import numpy as np
import matplotlib.pyplot as plt

# Set seed so we get the same random numbers
np.random.seed(3)
# Number of inputs
N = 6
# Number of dimensions of each input
D = 8
# Create an empty list
X = np.random.normal(size=(D, N))
# Print X
print(X)

# Number of heads
H = 2
# QDV dimension
H_D = int(D / H)

# Set seed so we get the same random numbers
np.random.seed(0)

# Choose random values for the parameters for the first head
omega_q1 = np.random.normal(size=(H_D, D))
omega_k1 = np.random.normal(size=(H_D, D))
omega_v1 = np.random.normal(size=(H_D, D))
beta_q1 = np.random.normal(size=(H_D, 1))
beta_k1 = np.random.normal(size=(H_D, 1))
beta_v1 = np.random.normal(size=(H_D, 1))

# Choose random values for the parameters for the second head
omega_q2 = np.random.normal(size=(H_D, D))
omega_k2 = np.random.normal(size=(H_D, D))
omega_v2 = np.random.normal(size=(H_D, D))
beta_q2 = np.random.normal(size=(H_D, 1))
beta_k2 = np.random.normal(size=(H_D, 1))
beta_v2 = np.random.normal(size=(H_D, 1))

# Choose random values for the parameters
omega_c = np.random.normal(size=(D, D))


# Define softmax operation that works independently on each column
def softmax_cols(data_in):
    # Exponentiate all of the values
    exp_values = np.exp(data_in)
    # Sum over columns
    denom = np.sum(exp_values, axis=0)
    # Compute softmax (numpy broadcasts denominator to all rows automatically)
    softmax = exp_values / denom
    # return the answer
    return softmax


# Now let's compute self attention in matrix form
def multihead_scaled_self_attention(
    X,
    omega_v1,
    omega_q1,
    omega_k1,
    beta_v1,
    beta_q1,
    beta_k1,
    omega_v2,
    omega_q2,
    omega_k2,
    beta_v2,
    beta_q2,
    beta_k2,
    omega_c,
):
    X_prime = np.zeros_like(X)

    Q1 = omega_q1 @ X + beta_q1
    K1 = omega_k1 @ X + beta_k1
    V1 = omega_v1 @ X + beta_v1

    KQ1 = (K1.T @ Q1) / np.sqrt(K1.shape[0])
    att1 = softmax_cols(KQ1)

    Q2 = omega_q2 @ X + beta_q2
    K2 = omega_k2 @ X + beta_k2
    V2 = omega_v2 @ X + beta_v2

    KQ2 = (K2.T @ Q2) / np.sqrt(K2.shape[0])
    att2 = softmax_cols(KQ2)

    # Shape of KQN is D/H x N. Vertically concatenate them into a single
    # matrix of shape D x N.
    cc = np.concatenate((V1 @ att1, V2 @ att2))
    X_prime = omega_c @ cc

    return X_prime


# Run the self attention mechanism
X_prime = multihead_scaled_self_attention(
    X,
    omega_v1,
    omega_q1,
    omega_k1,
    beta_v1,
    beta_q1,
    beta_k1,
    omega_v2,
    omega_q2,
    omega_k2,
    beta_v2,
    beta_q2,
    beta_k2,
    omega_c,
)

# Print out the results
np.set_printoptions(precision=3)
print("Your answer:")
print(X_prime)

print("True values:")
print("[[-21.207  -5.373 -20.933  -9.179 -11.319 -17.812]")
print(" [ -1.995   7.906 -10.516   3.452   9.863  -7.24 ]")
print(" [  5.479   1.115   9.244   0.453   5.656   7.089]")
print(" [ -7.413  -7.416   0.363  -5.573  -6.736  -0.848]")
print(" [-11.261  -9.937  -4.848  -8.915 -13.378  -5.761]")
print(" [  3.548  10.036  -2.244   1.604  12.113  -2.557]")
print(" [  4.888  -5.814   2.407   3.228  -4.232   3.71 ]")
print(" [  1.248  18.894  -6.409   3.224  19.717  -5.629]]")
