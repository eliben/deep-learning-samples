# Sample of calculating the sigmoid function and its derivative using radgrad.
#
# To see how the derivative function works on Numpy arrays, run this sample with
# matplotlib installed:
#
#   PYTHONPATH=. uv run --with matplotlib examples/sigmoid.py
import radgrad.numpy_wrapper as np
from radgrad import grad


def sigm(x):
    return 1 / (1 + np.exp(-x))


print(sigm(0.5))

dsigm_dx = grad(sigm)
print(dsigm_dx(0.5))

try:
    import numpy  # This is the actual Numpy, not radgrad's wrapper
    import matplotlib.pyplot as plt

    x = numpy.linspace(-4, 4, 1000)

    y = sigm(x)
    dy = dsigm_dx(x)[0]

    plt.plot(x, y, label="sigm")
    plt.plot(x, dy, label="dsigm_dx")
    plt.legend()
    plt.show()
except ImportError:
    print("Please install numpy and matplotlib to plot the function.")
