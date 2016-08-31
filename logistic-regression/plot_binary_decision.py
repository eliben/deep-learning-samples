# Helper code to plot a binary decision region.
#
# Eli Bendersky (http://eli.thegreenplace.net)
# This code is in the public domain
from __future__ import print_function
import argparse
import matplotlib.pyplot as plt
import numpy as np
from timer import Timer
import sys


if __name__ == '__main__':
    # Note: if we flip all values here we get the same intersection.
    theta = np.array([[-4], [0.5], [1]])

    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    xs = np.linspace(-4, 8, 200)
    ys = np.linspace(-4, 8, 200)
    xsgrid, ysgrid = np.meshgrid(xs, ys)
    plane = np.zeros_like(xsgrid)
    for i in range(xsgrid.shape[0]):
        for j in range(xsgrid.shape[1]):
            plane[i, j] = np.array([1, xsgrid[i, j], ysgrid[i, j]]).dot(theta)
    ax.contour(xsgrid, ysgrid, plane, levels=[0])
    ax.grid(True)
    ax.annotate(r'here $\hat{y}(x) > 0$', xy=(4, 4), fontsize=20)
    ax.annotate(r'here $\hat{y}(x) < 0$', xy=(0, 0), fontsize=20)

    fig.savefig('line.png', dpi=80)
    plt.show()
