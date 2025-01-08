# Basic reverse-mode automatic differentiation in Python. Create expressions
# out of Var objects, then call grad() on the final result to backpropagate.
# See the examples in the __main__ block at the end of the file.
#
# Eli Bendersky (https://eli.thegreenplace.net)
# This code is in the public domain
import math
import numbers
from collections import namedtuple

predecessor = namedtuple("predecessor", ["multiplier", "var"])


def is_number(v):
    return isinstance(v, numbers.Number)


class Var:
    def __init__(self, v):
        self.v = v
        self.predecessors = []
        self.gv = 0.0

    def __add__(self, other):
        if is_number(other):
            other = Var(other)
        out = Var(self.v + other.v)
        out.predecessors.append(predecessor(1.0, self))
        out.predecessors.append(predecessor(1.0, other))
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        out = Var(-self.v)
        out.predecessors.append(predecessor(-1.0, self))
        return out

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        if is_number(other):
            other = Var(other)
        out = Var(self.v * other.v)
        out.predecessors.append(predecessor(other.v, self))
        out.predecessors.append(predecessor(self.v, other))
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if is_number(other):
            other = Var(other)
        out = Var(self.v / other.v)
        out.predecessors.append(predecessor(1.0 / other.v, self))
        out.predecessors.append(predecessor(-self.v / (other.v**2), other))
        return out

    def __rtruediv__(self, other):
        if is_number(other):
            out = Var(other / self.v)
            out.predecessors.append(predecessor(-other / (self.v**2), self))
            return out
        else:
            raise NotImplementedError

    def grad(self, gv):
        self.gv += gv
        for p in self.predecessors:
            p.var.grad(p.multiplier * gv)


def exp(x):
    """e^x"""
    if is_number(x):
        x = Var(x)
    out = Var(math.exp(x.v))
    out.predecessors.append(predecessor(math.exp(x.v), x))
    return out


def ln(x):
    """ln(x)"""
    if is_number(x):
        x = Var(x)
    out = Var(math.log(x.v))
    out.predecessors.append(predecessor(1.0 / x.v, x))
    return out


def sin(x):
    """sin(x)"""
    if is_number(x):
        x = Var(x)
    out = Var(math.sin(x.v))
    out.predecessors.append(predecessor(math.cos(x.v), x))
    return out


if __name__ == "__main__":
    # Examples of expressions created our of Var notes and operations they
    # support, followed by calling grad(). After calling grad(), the
    # participating Vars should not be reused.
    x = Var(2.0)
    y = Var(5.0)
    z = 6 * x * y - x * y + x

    z.grad(1.0)
    print(x.gv, y.gv)

    xx = Var(0.5)
    sigmoid = 1 / (1 + exp(-xx))
    sigmoid.grad(1.0)
    print(xx.gv)

    # Example from the paper "Automatic Differentiation in Machine Learning:
    # a Survey" by Baydin et al.
    x1 = Var(2)
    x2 = Var(5)
    f = ln(x1) + x1 * x2 - sin(x2)
    f.grad(1.0)
    print(x1.gv, x2.gv)

    # More complicated variation on the previous example
    x1 = Var(2)
    x2 = Var(5)
    f = ln(x1) + x1 * x2 - sin(x1 * x2)
    f.grad(1.0)
    print(x1.gv, x2.gv)
