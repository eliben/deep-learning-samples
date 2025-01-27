# Plots a function twice (second time after the first plot is closed)

import numpy as np
import matplotlib.pyplot as plt


# Define the Rectified Linear Unit (ReLU) function
def ReLU(preactivation):
    activation = preactivation.clip(0.0)
    return activation


# Define a shallow neural network with, one input, one output, and three hidden units
def shallow_1_1_3(
    x,
    activation_fn,
    phi_0,
    phi_1,
    phi_2,
    phi_3,
    theta_10,
    theta_11,
    theta_20,
    theta_21,
    theta_30,
    theta_31,
):
    # Initial lines
    pre_1 = theta_10 + theta_11 * x
    pre_2 = theta_20 + theta_21 * x
    pre_3 = theta_30 + theta_31 * x
    # Activation functions
    act_1 = activation_fn(pre_1)
    act_2 = activation_fn(pre_2)
    act_3 = activation_fn(pre_3)
    # Weight activations
    w_act_1 = phi_1 * act_1
    w_act_2 = phi_2 * act_2
    w_act_3 = phi_3 * act_3
    # Combine weighted activation and add y offset
    y = phi_0 + w_act_1 + w_act_2 + w_act_3
    # Return everything we have calculated
    return y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3


def plot_neural(x, y):
    fig, ax = plt.subplots()
    ax.plot(x.T, y.T)
    ax.set_xlabel("Input")
    ax.set_ylabel("Output")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_aspect(1.0)
    plt.show()


# Now lets define some parameters and run the first neural network
n1_theta_10 = 0.0
n1_theta_11 = -1.0
n1_theta_20 = 0
n1_theta_21 = 1.0
n1_theta_30 = -0.67
n1_theta_31 = 1.0
n1_phi_0 = 1.0
n1_phi_1 = -2.0
n1_phi_2 = -3.0
n1_phi_3 = 9.3

# Define a range of input values
n1_in = np.arange(-1, 1, 0.01).reshape([1, -1])

# We run the neural network for each of these input values
n1_out, *_ = shallow_1_1_3(
    n1_in,
    ReLU,
    n1_phi_0,
    n1_phi_1,
    n1_phi_2,
    n1_phi_3,
    n1_theta_10,
    n1_theta_11,
    n1_theta_20,
    n1_theta_21,
    n1_theta_30,
    n1_theta_31,
)
# And then plot it
plot_neural(n1_in, n1_out)

# Now we'll define the same neural network, but this time, we will use matrix
# form as in equation 4.15. When you get this right, it will draw the same plot
# as above.

beta_0 = np.zeros((3, 1))
Omega_0 = np.zeros((3, 1))
beta_1 = np.zeros((1, 1))
Omega_1 = np.zeros((1, 3))

beta_0[0, 0] = n1_theta_10
beta_0[1, 0] = n1_theta_20
beta_0[2, 0] = n1_theta_30
Omega_0[0, 0] = n1_theta_11
Omega_0[1, 0] = n1_theta_21
Omega_0[2, 0] = n1_theta_31

beta_1[0, 0] = n1_phi_0
Omega_1[0, 0] = n1_phi_1
Omega_1[0, 1] = n1_phi_2
Omega_1[0, 2] = n1_phi_3

# Make sure that input data matrix has different inputs in its columns
n_data = n1_in.size
n_dim_in = 1
n1_in_mat = np.reshape(n1_in, (n_dim_in, n_data))

# This runs the network for ALL of the inputs, x at once so we can draw graph
h1 = ReLU(beta_0 + np.matmul(Omega_0, n1_in_mat))
n1_out = beta_1 + np.matmul(Omega_1, h1)

# Draw the network and check that it looks the same as the non-matrix case
plot_neural(n1_in, n1_out)
