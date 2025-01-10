from dataclasses import dataclass
from collections.abc import Callable
import typing
import numpy as np


@dataclass
class Node:
    vjp_func: Callable
    predecessors: list["Node"]


def make_root_node():
    return Node(None, [])


@dataclass
class Box:
    value: typing.Any
    node: Node


def maybe_box(value):
    if isinstance(value, Box):
        return value
    return Box(value=value, node=make_root_node())


def wrap_primitive(f):
    def wrapped(*args):
        # If no arguments are boxes, there's no tracing to be done. Just
        # call the primitive and return its result.
        if not any(isinstance(x, Box) for x in args):
            return f(*args)

        # For uniform handling in the rest of the function, make sure that
        # all inputs are boxes.
        # TODO: is this needed??? for constants??
        boxes = [maybe_box(x) for x in args]

        # Unbox the values, compute forward output and get obtain the
        # VJP function for this computation.
        output, vjp_func = vjp_rules[f](*[b.value for b in boxes])

        # Box the output and return it, with an associated Node.
        return Box(
            value=output,
            node=Node(vjp_func=vjp_func, predecessors=[b.node for b in boxes]),
        )

    return wrapped


add = wrap_primitive(np.add)
mul = wrap_primitive(np.multiply)
div = wrap_primitive(np.divide)
Box.__add__ = Box.__radd__ = add
Box.__mul__ = Box.__rmul__ = mul
Box.__sub__ = lambda self, other: self + neg(other)
Box.__rsub__ = lambda self, other: other + neg(self)
Box.__truediv__ = div
Box.__rtruediv__ = lambda self, other: div(other, self)
neg = Box.__neg__ = wrap_primitive(np.negative)
sin = wrap_primitive(np.sin)
cos = wrap_primitive(np.cos)
log = wrap_primitive(np.log)
exp = wrap_primitive(np.exp)

# vjp_rules holds the calculation and VJP rules for each primitive.
# (VJP = Vector-Jacobian Product)
# Structure:
#   vjp_rules[primitive] = (f, vjp)
#     primitive: The primitive function we've wrapped.
#     f: The underlying function that calculates the primitive. It should be
#        given unboxed arguments.
#     vjp: The function that calculates the vector-jacobian product for this
#          primitive. It takes the output gradient and returns input gradients
#          for each argument, as a list.
vjp_rules = {}
vjp_rules[np.add] = lambda x, y: (x + y, lambda g: [g, g])
vjp_rules[np.multiply] = lambda x, y: (x * y, lambda g: [y * g, x * g])
vjp_rules[np.sin] = lambda x: (np.sin(x), lambda g: [np.cos(x) * g])
vjp_rules[np.cos] = lambda x: (np.cos(x), lambda g: [-np.sin(x) * g])
vjp_rules[np.log] = lambda x: (np.log(x), lambda g: [g / x])
vjp_rules[np.exp] = lambda x: (np.exp(x), lambda g: [np.exp(x) * g])
vjp_rules[np.negative] = lambda x: (np.negative(x), lambda g: [-g])
vjp_rules[np.divide] = lambda x, y: (x / y, lambda g: [g / y, -g * x / y**2])


def backprop(arg_nodes, out_node, out_g):
    grads = {id(out_node): out_g}
    for node in toposort(out_node):
        g = grads.pop(id(node))

        inputs_g = node.vjp_func(g)
        print(f"Node: {node}, g={g}, inputs_g={inputs_g}")
        assert len(inputs_g) == len(node.predecessors)
        for inp_node, g in zip(node.predecessors, inputs_g):
            grads[id(inp_node)] = grads.get(id(inp_node), 0.0) + g
            print(f"  set {inp_node} to {grads[id(inp_node)]}")
    return [grads.get(id(node), 0.0) for node in arg_nodes]


def toposort(out_node):
    """Topological sort of the computation graph starting at out_node.

    Yields nodes in topologically sorted order.
    """
    visited = set()

    def postorder(node):
        visited.add(id(node))
        for pred in node.predecessors:
            if not id(pred) in visited:
                yield from postorder(pred)
        yield node

    return reversed([node for node in postorder(out_node) if node.predecessors])


import inspect


def grad(f):
    def wrapped(*args):
        boxed_args = [Box(value=x, node=make_root_node()) for x in args]
        out = f(*boxed_args)
        arg_nodes = [b.node for b in boxed_args]

        for n in toposort(out.node):
            print(f"- {n}")
            print(f"  {inspect.getsource(n.vjp_func)}")

        return backprop(arg_nodes, out.node, 1.0)

    return wrapped


if __name__ == "__main__":

    # def f(x):
    #     return sin(x) + x

    # print(f(2))
    # fg = grad(f)
    # print(fg(2))

    # def f(x1, x2):
    #     return log(x1) + x1 * x2 - sin(x2)

    # print(f(2, 5))
    # fg = grad(f)
    # print(fg(2, 5))

    def sigm(x):
        return 1 / (1 + exp(-x))

    print(sigm(0.5))

    fg = grad(sigm)
    print(fg(0.5))
