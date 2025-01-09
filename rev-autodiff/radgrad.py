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
Box.__add__ = Box.__radd__ = add
Box.__mul__ = Box.__rmul__ = mul
sin = wrap_primitive(np.sin)
cos = wrap_primitive(np.cos)

# vjp_rules holds the calculation and gradient rules for each primitive.
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


def backward_pass(arg_nodes, out_node, output_grad):
    grads = {id(out_node): output_grad}


def grad(f):
    def wrapped(*args):
        boxed_args = [Box(value=x, node=make_root_node()) for x in args]
        out = f(*boxed_args)
        arg_nodes = [b.node for b in boxed_args]
        return backward_pass(arg_nodes, out.node, 1.0)

    return wrapped


if __name__ == "__main__":

    def f(x):
        return sin(x) + x

    print(f(3))

    fg = grad(f)
    print(fg(3))
