# Helper code to plot binary losses.
#
# Eli Bendersky (http://eli.thegreenplace.net)
# This code is in the public domain
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    xs = np.linspace(-2, 2, 500)

    # plot L0/1 loss
    ax.plot(xs, np.where(xs < 0, np.ones_like(xs), np.zeros_like(xs)),
            color='r', linewidth=2.0, label='$L_{01}$')

    # plot square loss
    ax.plot(xs, (xs - 1) ** 2, linestyle='-.', label='$L_2$')

    # plot hinge loss
    ax.plot(xs, np.maximum(np.zeros_like(xs), 1 - xs),
            color='g', linewidth=2.0, label='$L_h$')

    ax.grid(True)

    plt.ylim((-1, 4))
    ax.legend()

    fig.savefig('loss.png', dpi=80)
    plt.show()
