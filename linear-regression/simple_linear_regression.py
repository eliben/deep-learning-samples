from __future__ import print_function
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np


def generate_data(n, m=2.25, b=6.0, stddev=1.5):
    """Generate n data points approximating given line.

    m, b: line slope and intercept
    stddev: standard deviation of added error
    """
    x = np.linspace(-2.0, 2.0, n)
    y = x * m + b + np.random.normal(loc=0, scale=stddev, size=n)
    return x, y


def plot_scatter_data(x, y):
    plt.scatter(x, y, marker='x')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def compute_cost(x, y, m, b):
    """Compute the MSE cost of a prediction based on m, b.

    x: inputs vector
    y: actual outputs vector
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
    n = x.shape[0]
    m, b = 0, 0

    for step in range(nsteps):
        yield m, b, compute_cost(x, y, m, b)
        yhat = m * x + b
        diff = yhat - y
        dm = learning_rate * (diff * x).sum() * 2 / n
        db = learning_rate * diff.sum() * 2 / n
        m -= dm
        b -= db

    yield m, b, compute_cost(x, y, m, b)


def plot_cost_3D(x, y, costfunc):
    """Plot cost as 3D and contour.

    x, y: x and y values from the dataset
    costfunc: cost function with signature like compute_cost
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
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    ax.set_xlabel('b')
    ax.set_ylabel('m')
    msgrid, bsgrid = np.meshgrid(ms, bs)
    surf = ax.plot_surface(msgrid, bsgrid, cost, cmap=cm.coolwarm)

    # Configure contour plot.
    plt.figure(2)
    plt.contour(msgrid, bsgrid, cost)
    plt.xlabel('b')
    plt.ylabel('m')
    plt.show()


def plot_cost_vs_step(costs):
    plt.plot(range(len(costs)), costs)
    plt.show()


if __name__ == '__main__':
    # For reproducibility
    np.random.seed(42)

    N = 500
    x, y = generate_data(N)
    #plot_scatter_data(x, y)
    #plot_cost_3D(x, y, compute_cost)

    #print(compute_cost(x, y, m=40, b=10))

    costs = [c for _, _, c in gradient_descent(x, y, 50)]
    plot_cost_vs_step(costs)
    #for m, b, cost in gradient_descent(x, y, 50):
        #print(m, b, cost)
