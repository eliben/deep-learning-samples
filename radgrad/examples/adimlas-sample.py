# Sample calculation from "Automatic Differentiation in Machine Learning: a
# Survey" by Baydin et al.

import radgrad.numpy_wrapper as np
from radgrad import grad


def f(x1, x2):
    return np.log(x1) + x1 * x2 - np.sin(x2)


print(f(2, 5))

df_dx = grad(f)
print(df_dx(2, 5))
