import numpy as np
import matplotlib.pyplot as plt


# Define a signal that we can apply convolution to
x = np.array([5.2, 5.3, 5.4, 5.1, 10.1, 10.3, 9.9, 10.3, 3.2, 3.4, 3.3, 3.1])


# Draw the signal
# fig, ax = plt.subplots()
# ax.plot(x, "k-")
# ax.set_xlim(0, 11)
# ax.set_ylim(0, 12)
# plt.show()


# Now let's define a zero-padded convolution operation
# with a convolution kernel size of 3, a stride of 1, and a dilation of 1
# as in figure 10.2a-c.  Write it yourself, don't call a library routine!
# Don't forget that Python arrays are indexed from zero, not from 1 as in the book figures
def conv_3_1_1_zp(x_in, omega):
    x_out = np.zeros_like(x_in)

    # padded is x_in, with zeros on both ends
    padded = np.zeros(len(x_in) + 2)
    padded[1:-1] = x_in

    for i in range(len(x_out)):
        x_out[i] = np.sum(padded[i : i + 3] * omega)

    return x_out


omega = np.array([0.33, 0.33, 0.33])
h = conv_3_1_1_zp(x, omega)

# Check that you have computed this correctly
print(f"Sum of output is {np.sum(h):3.3}, should be 71.1")

# Draw the signal
# fig, ax = plt.subplots()
# ax.plot(x, "k-", label="before")
# ax.plot(h, "r-", label="after")
# ax.set_xlim(0, 11)
# ax.set_ylim(0, 12)
# ax.legend()
# plt.show()

# omega = np.array([-0.5, 0, 0.5])
# h2 = conv_3_1_1_zp(x, omega)

# # Draw the signal
# fig, ax = plt.subplots()
# ax.plot(x, "k-", label="before")
# ax.plot(h2, "r-", label="after")
# ax.set_xlim(0, 11)
# # ax.set_ylim(0, 12)
# ax.legend()
# plt.show()
