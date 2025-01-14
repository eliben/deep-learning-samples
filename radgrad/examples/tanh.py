import radgrad.numpy_wrapper as np
from radgrad import grad


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


print(tanh(1.0))

dtanh_dx = grad(tanh)
print(dtanh_dx(1.0))
