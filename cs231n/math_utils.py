import numpy as np
from random import randrange

def rel_error(x, y):
    """Relative error between x and y."""
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def grad_check_sparse(f, x, analytic_grad, num_checks):
    """Run some checks of the computed analytic gradient vs. numeric gradient.
    """
    h = 1e-5

    for i in xrange(num_checks):
        ix = tuple([randrange(m) for m in x.shape])

        x[ix] += h  # increment by h
        fxph = f(x)  # evaluate f(x + h)
        x[ix] -= 2 * h  # increment by h
        fxmh = f(x)  # evaluate f(x - h)
        x[ix] += h  # reset

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / (
            abs(grad_numerical) + abs(grad_analytic))
        print 'numerical: %f analytic: %f, relative error: %e' % (
            grad_numerical, grad_analytic, rel_error)
