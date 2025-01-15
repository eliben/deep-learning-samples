from dataclasses import dataclass
from collections.abc import Callable
import typing
import numpy as _np


@dataclass
class Node:
    vjp_func: Callable
    predecessors: list["Node"]


def make_root_node():
    """Empty node with no predecessors."""
    return Node(None, [])


@dataclass
class Box:
    value: typing.Any
    node: Node


def maybe_box(value):
    """Box the value if it's not already a Box."""
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
        boxes = [maybe_box(x) for x in args]

        # Unbox the values, compute forward output and obtain the
        # VJP function for this computation.
        output, vjp_func = vjp_rules[f](*[b.value for b in boxes])

        # Box the output and return it, with an associated Node.
        return Box(
            value=output,
            node=Node(vjp_func=vjp_func, predecessors=[b.node for b in boxes]),
        )

    return wrapped


# vjp_rules holds the calculation and VJP rules for each primitive.
# (VJP = Vector-Jacobian Product)
# Structure:
#   vjp_rules[primitive] = maker_func(*args)
#     primitive: The primitive function we've wrapped.
#     maker_func(*args):
#       takes the runtime values of arguments passed into the primitive and
#       returns a tuple (output, vjp_func). The output is the result of the
#       forward computation of the primitive with *args, and vjp_func
#       calculates the vector-jacobian product. It takes the output gradient
#       and returns input gradients of the primitive for each argument,
#       as a list.
vjp_rules = {}


def add_vjp_rule(np_primitive, vjp_maker_func):
    vjp_rules[np_primitive] = vjp_maker_func


def backprop(arg_nodes, out_node, out_g):
    grads = {id(out_node): out_g}
    for node in toposort(out_node):
        g = grads.pop(id(node))

        inputs_g = node.vjp_func(g)
        # print(f"Node: {node}, g={g}, inputs_g={inputs_g}")
        assert len(inputs_g) == len(node.predecessors)
        for inp_node, g in zip(node.predecessors, inputs_g):
            grads[id(inp_node)] = grads.get(id(inp_node), 0.0) + g
            # print(f"  set {inp_node} to {grads[id(inp_node)]}")
    return [grads.get(id(node), 0.0) for node in arg_nodes]


def toposort(out_node):
    """Topological sort of the computation graph starting at out_node.

    Yields nodes in topologically sorted order.
    """
    visited = set()

    def postorder(node):
        visited.add(id(node))
        for pred in node.predecessors:
            if id(pred) not in visited:
                yield from postorder(pred)
        yield node

    return reversed([node for node in postorder(out_node) if node.predecessors])


def grad(f):
    def wrapped(*args):
        boxed_args = [Box(value=x, node=make_root_node()) for x in args]
        out = f(*boxed_args)
        arg_nodes = [b.node for b in boxed_args]

        # import inspect
        # for n in toposort(out.node):
        #     print(f"- {n}")
        #     print(f"  {inspect.getsource(n.vjp_func)}")

        return backprop(arg_nodes, out.node, _np.float64(1.0))

    return wrapped
