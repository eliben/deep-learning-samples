import radgrad.numpy_wrapper as np
from radgrad import grad


def sigm(x):
    return 1 / (1 + np.exp(-x))


print(sigm(0.5))

dsigm_dx = grad(sigm)
print(dsigm_dx(0.5))
