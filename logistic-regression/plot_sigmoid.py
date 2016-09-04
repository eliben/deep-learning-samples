# Helper code to plot the sigmoid function.
#
# Eli Bendersky (http://eli.thegreenplace.net)
# This code is in the public domain
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np

from regression_lib import sigmoid


if __name__ == '__main__':
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    x = np.linspace(-6, 6, 200)
    y = sigmoid(x)

    ax.plot(x, y, 'b-', linewidth=2)
    ax.grid(True)

    fig.savefig('sigmoid.png', dpi=80)
    plt.show()
