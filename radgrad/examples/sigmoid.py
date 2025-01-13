# Run this sample with matplotlib installed:
s  #
#   PYTHONPATH=. uv run --with matplotlib examples/sigmoid.py
import radgrad.numpy_wrapper as np
from radgrad import grad


def sigm(x):
    return 1 / (1 + np.exp(-x))


print(sigm(0.5))

dsigm_dx = grad(sigm)
print(dsigm_dx(0.5))

try:
    import numpy

    x = numpy.linspace(-4, 4, 1000)

    y = sigm(x)
    dy = dsigm_dx(x)[0]

    import matplotlib.pyplot as plt

    plt.plot(x, y)
    plt.plot(x, dy)
    plt.show()
except ImportError:
    print("Please install numpy and matplotlib to plot the function.")
