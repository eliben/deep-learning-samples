import numpy as _np
from .radgrad import wrap_primitive, Box

add = wrap_primitive(_np.add)
mul = wrap_primitive(_np.multiply)
div = wrap_primitive(_np.divide)
neg = wrap_primitive(_np.negative)
sin = wrap_primitive(_np.sin)
cos = wrap_primitive(_np.cos)
log = wrap_primitive(_np.log)
exp = wrap_primitive(_np.exp)
abs = wrap_primitive(_np.fabs)

Box.__add__ = Box.__radd__ = add
Box.__mul__ = Box.__rmul__ = mul
Box.__sub__ = lambda self, other: self + neg(other)
Box.__rsub__ = lambda self, other: other + neg(self)
Box.__truediv__ = div
Box.__rtruediv__ = lambda self, other: div(other, self)
Box.__neg__ = neg
Box.__abs__ = abs
