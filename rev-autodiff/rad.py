import math
from collections import namedtuple

predecessors = namedtuple('predecessors', ['multiplier', 'var'])

# TODO: what about constants??

class Var:
    def __init__(self, v):
        self.v = v
        self.predecessors = []
        self.gv = 0.0

    def __add__(self, other):
        out = Var(self.v + other.v)
        out.predecessors.append(predecessors(1.0, self))
        out.predecessors.append(predecessors(1.0, other))
        return out

    def __mul__(self, other):
        out = Var(self.v * other.v)
        out.predecessors.append(predecessors(other.v, self))
        out.predecessors.append(predecessors(self.v, other))
        return out
    
    def grad(self, gv):
        self.gv += gv
        for p in self.predecessors:
            p.var.grad(p.multiplier * gv)


if __name__ == '__main__':
    x = Var(2.0)
    y = Var(5.0)
    z = x * y + x * y + x

    z.grad(1.0)

    print(x.gv, y.gv)


