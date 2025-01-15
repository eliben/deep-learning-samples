import numpy as _np
from .radgrad import wrap_primitive, Box, add_vjp_rule

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
Box.__abs__ = lambda self: neg(self)

add_vjp_rule(_np.add, lambda x, y: (x + y, lambda g: [g, g]))
add_vjp_rule(_np.multiply, lambda x, y: (x * y, lambda g: [y * g, x * g]))
add_vjp_rule(_np.sin, lambda x: (_np.sin(x), lambda g: [_np.cos(x) * g]))
add_vjp_rule(_np.cos, lambda x: (_np.cos(x), lambda g: [-_np.sin(x) * g]))
add_vjp_rule(_np.log, lambda x: (_np.log(x), lambda g: [g / x]))
add_vjp_rule(_np.exp, lambda x: (_np.exp(x), lambda g: [_np.exp(x) * g]))
add_vjp_rule(_np.negative, lambda x: (_np.negative(x), lambda g: [-g]))
add_vjp_rule(_np.divide, lambda x, y: (x / y, lambda g: [g / y, -g * x / (y * y)]))
