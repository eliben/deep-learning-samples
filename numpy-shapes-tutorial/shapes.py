import numpy as np

# One-dimensional array will have ndim=1, and shape (5,)
# Passing in the tuple (5,) is the same as passing in 5
aa1 = np.random.rand(5)
print("aa1", aa1.ndim, aa1.shape, aa1)

# Two-dimensional array will have ndim=2, and shape (3, 4)
aa2 = np.random.rand(3, 4)
print("aa2", aa2.ndim, aa2.shape, aa2)

# The shape is always a tuple
# A scalar is a zero-dimensional array, e.g. np.shape(1) will return ()

# Dimensions are also called "axes". This array has 3 axes, the first with
# size 2, the second with size 3, and the third with size 4
aa3 = np.random.rand(2, 3, 4)
print("aa3", aa3.ndim, aa3.shape, aa3)

# When Numpy arrays are printed, their data order is used; the default order is
# "C", which means row-major. So the last dimension is changing the fastest (and
# thus will be shown as contiguous in the output). This is only the physical
# layout/representation, though; logically it makes no difference.
# The concept of "rows" and "columns" is just a convention. Notice how aa2 is
# printed out. the size of axis 0 is 3, the size of axis 1 is 4. So we see it
# as having 3 "rows", each with 4 "columns".

# Indexing order is the same as the shape order; this is the "last" element
# in aa3 because it's last in every dimension
print("last in aa3:", aa3[1, 2, 3])

# This will throw an error because the first index is out of bounds
# print("last in aa3:", aa3[2, 2, 3])

# Dimensions of size 1 are valid, but the only valid addressing for them is
# the 0th index.
#
aa14 = np.random.rand(1, 4)
print("aa14", aa14.ndim, aa14.shape, aa14)
aa41 = np.random.rand(4, 1)
print("aa41", aa41.ndim, aa41.shape, aa41)

# dot for 1d arrays is the same as inner product, we can use the dot method,
# the dot function or the @ operator.
bb1 = np.random.rand(5)
print(np.dot(aa1, bb1))
print(aa1.dot(bb1))
print(aa1 @ bb1)

# When the arrays are 2d, the dot function and @ operator do matrix
# multiplication. If LHS has shape (m,n), then RHS must have shape (n,p) for
# any p and the result is shape (m,p).
bb2 = np.random.rand(4, 5)
dotres = aa2 @ bb2
print("dotres:", dotres.shape, dotres)

# Therefore, an outer product is achieved by multiplying a (n,1) array by
# an (1,m) array.
outer = np.random.randn(3, 1) @ np.random.randn(1, 2)
print("outer:", outer.shape, outer)

# Dot between a matrix and a vector will produce a 1d array ("row" vector)
mat = np.ones((3, 4))
print("mat @ v:", mat @ np.arange(4))

# If we want a column vector, we have to explicitly reshape the RHS vector
# to a column vector.
print("mat @ v col:", mat @ np.arange(4).reshape(4, 1))

# LHS vector
print("v @ mat:", np.arange(3) @ mat)
