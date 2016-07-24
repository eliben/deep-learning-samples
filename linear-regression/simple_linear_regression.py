from __future__ import print_function
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def generate_data(n, m=2.25, b=6.0, stddev=1.5):
    """Generate n data points approximating given line.

    m, b: line slope and intercept
    stddev: standard deviation of added error
    """
    x = np.linspace(-2.0, 2.0, n)
    y = x * m + b + np.random.normal(loc=0, scale=stddev, size=n)
    return x, y


def plot_data(x, y, mb_history=None):
    """Plot the data: y as a function of x, in a scatterplot.

    x, y: arrays of data.
    mb_history:
        if provided, it's a sequence of (m, b) pairs that are used to draw
        animated lines on top of the scatterplot.
    """
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    ax.scatter(x, y, marker='x')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    if mb_history:
        m0, b0 = mb_history[0]
        line, = ax.plot(x, x * m0 + b0, 'r-', linewidth=2.0)

        def update(frame_i):
            mi, bi = mb_history[frame_i]
            line.set_ydata(x * mi + bi)
            ax.set_title('Fit at iteration {0}'.format(frame_i))
            return [line]

        anim = FuncAnimation(fig, update, frames=range(len(mb_history)),
                             interval=200)
    plt.show()


def compute_cost(x, y, m, b):
    """Compute the MSE cost of a prediction based on m, b.

    x: inputs vector
    y: observed outputs vector
    m, b: regression parameters

    Returns: a scalar cost.
    """
    yhat = m * x + b
    diff = yhat - y
    # Vectorized computation using a dot product to compute sum of squares.
    cost = np.dot(diff.T, diff) / float(x.shape[0])
    # Cost is a 1x1 matrix, we need a scalar.
    return cost.flat[0]


def gradient_descent(x, y, nsteps, learning_rate=0.1):
    """Runs gradient descent optimization to fit a line y^ = x * m + b.

    x, y: input data and observed outputs.
    nsteps: how many steps to run the optimization for.
    learning_rate: learning rate of gradient descent.

    Yields 'nsteps + 1' triplets of (m, b, cost) where m, b are the fit
    parameters for the given step, and cost is their cost vs the real y.
    """
    n = x.shape[0]
    # Start with m and b initialized to 0s for the first try.
    m, b = 0, 0
    yield m, b, compute_cost(x, y, m, b)

    for step in range(nsteps):
        yhat = m * x + b
        diff = yhat - y
        dm = learning_rate * (diff * x).sum() * 2 / n
        db = learning_rate * diff.sum() * 2 / n
        m -= dm
        b -= db
        yield m, b, compute_cost(x, y, m, b)


def plot_cost_3D(x, y, costfunc, mb_history=None):
    """Plot cost as 3D and contour.

    x, y: x and y values from the dataset
    costfunc: cost function with signature like compute_cost
    mb_history:
        if provided, it's a sequence of (m, b) pairs that are added as
        crosshairs markers on top of the contour plot.
    """
    lim = 10.0
    N = 250
    ms = np.linspace(-lim, lim, N)
    bs = np.linspace(-lim, lim, N)
    cost = np.zeros((N, N))
    for m_idx in range(N):
        for b_idx in range(N):
            cost[m_idx, b_idx] = costfunc(x, y, ms[m_idx], bs[b_idx])
    # Configure 3D plot.
    fig = plt.figure()
    fig.set_tight_layout(True)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.set_xlabel('b')
    ax1.set_ylabel('m')
    msgrid, bsgrid = np.meshgrid(ms, bs)
    surf = ax1.plot_surface(msgrid, bsgrid, cost, cmap=cm.coolwarm)

    # Configure contour plot.
    ax2 = fig.add_subplot(1, 2, 2)
    #plt.figure(2)
    ax2.contour(msgrid, bsgrid, cost)
    ax2.set_xlabel('b')
    ax2.set_ylabel('m')

    if mb_history:
        ms, bs = zip(*mb_history)
        plt.plot(bs, ms, 'rx', mew=3, ms=5)

    plt.show()


def plot_cost_vs_step(costs):
    plt.plot(range(len(costs)), costs)
    plt.show()


if __name__ == '__main__':
    # For reproducibility
    np.random.seed(42)

    N = 500
    x, y = generate_data(N)

    NSTEPS = 30
    mbcost = list(gradient_descent(x, y, NSTEPS))
    print(mbcost[-1])
    mb_history = [(m, b) for m, b, _ in mbcost]
    #plot_cost_3D(x, y, compute_cost, mb_history)
    #plot_data(x, y, mb_history)

    #costs = [c for _, _, c in gradient_descent(x, y, 50)]
    #plot_cost_vs_step([item[2] for item in mbcost])
