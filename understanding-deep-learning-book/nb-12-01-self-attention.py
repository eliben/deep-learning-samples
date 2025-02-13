import numpy as np
import matplotlib.pyplot as plt

# Set seed so we get the same random numbers
np.random.seed(3)
# Number of inputs
N = 3
# Number of dimensions of each input
D = 4
# Create an empty list
all_x = []
# Create elements x_n and append to list
for n in range(N):
    all_x.append(np.random.normal(size=(D, 1)))
# Print out the list
print(all_x)

np.random.seed(0)

# Attention parameters for Q, K, V
omega_q = np.random.normal(size=(D, D))
omega_k = np.random.normal(size=(D, D))
omega_v = np.random.normal(size=(D, D))
beta_q = np.random.normal(size=(D, 1))
beta_k = np.random.normal(size=(D, 1))
beta_v = np.random.normal(size=(D, 1))

# Make three lists to store queries, keys, and values
all_queries = []
all_keys = []
all_values = []
# For every input
for x in all_x:
    # Shapes: (D, 1) = (D, D) @ (D, 1) + (D, 1)
    query = omega_q @ x + beta_q
    key = omega_k @ x + beta_k
    value = omega_v @ x + beta_v

    all_queries.append(query)
    all_keys.append(key)
    all_values.append(value)


def softmax(x):
    """Compute the softmax of vector x."""
    exps = np.exp(x)
    return exps / np.sum(exps)


# Output
all_x_prime = []

# For each output
for n in range(N):
    # Create list for dot products of query N with all keys
    all_km_qn = []
    # Compute the dot products
    for key in all_keys:
        # TODO -- compute the appropriate dot product
        # Replace this line
        dot_product = all_queries[n].T @ key

        # Store dot product
        all_km_qn.append(dot_product[0][0])

    attention = softmax(all_km_qn)
    # Print result (should be positive sum to one)
    print("Attentions for output ", n)
    print(attention)

    print(f"shapes attention={attention.shape}, all_values[n]={all_values[n].shape}")

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
