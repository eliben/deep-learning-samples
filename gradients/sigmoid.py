from __future__ import print_function

import numpy as np
from numgrad import eval_numerical_gradient


def sigmoid(z):
    """Computes sigmoid function.

    z: array of input values.

    Returns array of outputs, sigmoid(z).
    """
    # Note: this version of sigmoid tries to avoid overflows in the computation
    # of e^(-z), by using an alternative formulation when z is negative, to get
    # 0. e^z / (1+e^z) is equivalent to the definition of sigmoid, but we won't
    # get e^(-z) to overflow when z is very negative.
    # Since both the x and y arguments to np.where are evaluated by Python, we
    # may still get overflow warnings for large z elements; therefore we ignore
    # warnings during this computation.
    with np.errstate(over='ignore', invalid='ignore'):
        return np.where(z >= 0,
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z)))

def sigmoid_grad(x):
    return sigmoid(x) * (1 - sigmoid(x))


if __name__ == '__main__':
    x = np.array([1.0, 2.1, 0.3, 0.7])
    print('sigmoid', sigmoid(x))
    print('sigmoid_grad', sigmoid_grad(x))

    # Note: eval_numerical_gradient works for scalar functions. Therefore we'll
    # run it for each element of sigmoid separately.
    print('Numerical gradient')
    for i in range(x.shape[0]):
        print(i, eval_numerical_gradient(lambda z: sigmoid(z)[i], x))
