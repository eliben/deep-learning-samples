from radgrad import grad
import math


def taylor_sin(x):
    ans = currterm = x
    for i in range(0, 50):
        currterm = -currterm * x * x / ((2 * i + 3) * (2 * i + 2))
        ans = ans + currterm
    return ans


dsin_dx = grad(taylor_sin)

for v in [0.0, math.pi / 4, math.pi / 2, math.pi]:
    print(f"sin({v:.3}) = {taylor_sin(v):.3}")
    print(f"dsin_dx({v:.3}) = {dsin_dx(v)[0]:.3}")

try:
    import numpy  # This is the actual Numpy, not radgrad's wrapper
    import matplotlib.pyplot as plt

    x = numpy.linspace(-3.3, 3.3, 1000)

    y = taylor_sin(x)
    dy = dsin_dx(x)[0]

    plt.plot(x, y, label="sin")
    plt.plot(x, dy, label="dsin_dx")
    plt.legend()
    plt.show()
except ImportError:
    print("Please install numpy and matplotlib to plot the function.")
