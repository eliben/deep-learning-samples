from __future__ import print_function

import numpy as np
from numgrad import eval_numerical_gradient


def tanh_grad(x):
    return 1 - np.tanh(x) ** 2


if __name__ == '__main__':
    x = np.array([1.0, 2.1, 0.3, 0.7])
    print('tanh', np.tanh(x))
    print('tanh_grad', tanh_grad(x))

    # Note: eval_numerical_gradient works for scalar functions. Therefore we'll
    # run it for each element of tanh separately.
    print('Numerical gradient')
    for i in range(x.shape[0]):
        print(i, eval_numerical_gradient(lambda z: np.tanh(z)[i], x))
