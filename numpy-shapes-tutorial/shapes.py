import numpy as np

# One-dimensional array will have ndim=1, and shape (5,)
aa1 = np.random.rand(5)
print("aa1", aa1.ndim, aa1.shape, aa1)

# Two-dimensional array will have ndim=2, and shape (3, 4)
aa2 = np.random.rand(3, 4)
print("aa2", aa2.ndim, aa2.shape, aa2)

# The shape is always a tuple
# A scalar is a zero-dimensional array, e.g. np.shape(1) will return ()

aa3 = np.random.rand(2, 3, 4)
print("aa3", aa3.ndim, aa3.shape, aa3)

# When Numpy arrays are printed, their order is used; the default order
# is "C", which means row-major. So the last dimension is changing the
# fastest (and thus will be shown as contiguous in the output).
