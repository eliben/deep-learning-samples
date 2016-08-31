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
    ax.plot(xs, np.where(xs < 0, np.ones_like(xs), np.zeros_like(xs)),
            color='r', linewidth=2.0)
    ax.grid(True)

    plt.ylim((-1, 4))

    fig.savefig('01loss.png', dpi=80)
    plt.show()
