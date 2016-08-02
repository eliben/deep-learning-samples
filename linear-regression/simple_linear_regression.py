# Example of solving simple linear (y(x) = mx + b) regression in Python.
#
# Uses only Numpy, with Matplotlib for plotting.
#
# Eli Bendersky (http://eli.thegreenplace.net)
# This code is in the public domain
from __future__ import print_function
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from timer import Timer


def generate_data(n, m=2.25, b=6.0, stddev=1.5):
    """Generate n data points approximating given line.

    m, b: line slope and intercept.
    stddev: standard deviation of added error.

    Returns pair x, y: arrays of length n.
    """
    x = np.linspace(-2.0, 2.0, n)
    y = x * m + b + np.random.normal(loc=0, scale=stddev, size=n)
    return x, y


def plot_data_scatterplot(x, y, mb_history=None):
    """Plot the data: y as a function of x, in a scatterplot.

    x, y: arrays of data.
    mb_history:
        if provided, it's a sequence of (m, b) pairs that are used to draw
        animated lines on top of the scatterplot.
    """
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    fig.set_size_inches((8, 6))
    save_dpi = 80

    ax.scatter(x, y, marker='x')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    if mb_history:
        m0, b0 = mb_history[0]
        line, = ax.plot(x, x * m0 + b0, 'r-', linewidth=2.0)

        # Downsample mb_history by 2 to reduce the number of frames shown.
        def update(frame_i):
            mi, bi = mb_history[frame_i * 2]
            line.set_ydata(x * mi + bi)
            ax.set_title('Fit at iteration {0}'.format(frame_i * 2))
            return [line]

        anim = FuncAnimation(fig, update, frames=range(len(mb_history) // 2),
                             interval=200)
        anim.save('regressionfit.gif', dpi=save_dpi, writer='imagemagick')
    else:
        fig.savefig('linreg-data.png', dpi=save_dpi)
    plt.show()


def compute_cost(x, y, m, b):
    """Compute the MSE cost of a prediction based on m, b.

    x: inputs array.
    y: observed outputs array.
    m, b: regression parameters.

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

    x, y: input data and observed outputs, as array.
    nsteps: how many steps to run the optimization for.
    learning_rate: learning rate of gradient descent.

    Yields 'nsteps + 1' triplets of (m, b, cost) where m, b are the fit
    parameters for the given step, and cost is their cost vs. the real y. The
    first triplet has the initial m, b and cost; the rest carry results after
    each of the iteration steps.
    """
    n = x.shape[0]
    # Start with m and b initialized to 0s for the first try.
    m, b = 0, 0
    yield m, b, compute_cost(x, y, m, b)
    for step in range(nsteps):
        # Update m and b following the formulae for gradient updates.
        yhat = m * x + b
        diff = yhat - y
        dm = learning_rate * (diff * x).sum() * 2 / n
        db = learning_rate * diff.sum() * 2 / n
        m -= dm
        b -= db
        yield m, b, compute_cost(x, y, m, b)


def plot_cost_3D(x, y, costfunc, mb_history=None):
    """Plot cost as 3D and contour.

    x, y: arrays of data.
    costfunc: cost function with signature like compute_cost.
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
    ax2.contour(msgrid, bsgrid, cost)
    ax2.set_xlabel('b')
    ax2.set_ylabel('m')

    if mb_history:
        ms, bs = zip(*mb_history)
        plt.plot(bs, ms, 'rx', mew=3, ms=5)

    plt.show()


def plot_cost_vs_step(costs):
    """Given an array of costs, plots them vs. index."""
    plt.plot(range(len(costs)), costs)
    plt.show()


def compute_mb_analytic(x, y):
    """Given arrays of x, y computes m, b analytically."""
    xbar = np.average(x)
    ybar = np.average(y)
    m = (xbar * ybar - np.average(x * y)) / (xbar ** 2 - np.average(x ** 2))
    b = ybar - m * xbar
    return m, b


def compute_rsquared(x, y, m, b):
    """Compute R^2 - the coefficient of determination for m, b.

    x, y: arrays of input, output.
    m, b: regression parameters.

    Returns the R^2 - a scalar.
    """
    yhat = m * x + b
    diff = yhat - y
    SE_line = np.dot(diff.T, diff)
    SE_y = len(y) * y.var()
    return 1 - SE_line / SE_y


if __name__ == '__main__':
    # Follow through the code here to see how the functions are used. No
    # plotting is done by default. Uncomment relevant lines to produce plots.

    # For reproducibility.
    np.random.seed(42)

    # Generate some pseudo-random data we're goign to fit with linear
    # regression.
    N = 500
    x, y = generate_data(N)
    print('Generated {0} data points'.format(N))

    # Run gradient descent.
    NSTEPS = 50
    with Timer('Running gradient descent [{0} steps]'.format(NSTEPS)):
        mbcost = list(gradient_descent(x, y, NSTEPS))
        mb_history = [(m, b) for m, b, _ in mbcost]

    print('Final m={0}, b={1}; cost={2}'.format(mbcost[-1][0], mbcost[-1][1],
                                                mbcost[-1][2]))

    # Plot the data in a scatterplot, with an animated line fit.
    #plot_data_scatterplot(x, y, mb_history)

    # Plot the cost function in 3D and as contours; add markers for the costs
    # values returned by the gradient descent procedure.
    #plot_cost_3D(x, y, compute_cost, mb_history)

    m, b = compute_mb_analytic(x, y)
    print('Analytic: m={0}, b={1}'.format(m, b))

    rsquared = compute_rsquared(x, y, m, b)
    print('Rsquared:', rsquared)
