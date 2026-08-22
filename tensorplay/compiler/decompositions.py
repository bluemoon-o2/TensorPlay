"""Operator decomposition pass (L5-M4).

Modeled on ``torch/_inductor/decomposition.py``: rewrite composite
operators into the primitive set *before* AOT, so the derivative registry
only has to cover primitives. Every expansion below uses ops that already
have local vector-Jacobian rules (mul/add/sub/truediv/neg/exp), which keeps
decomposed graphs differentiable by construction.
"""

from __future__ import annotations

import operator
from typing import Any, Callable, Dict, Optional, Tuple

from .graph import Graph, GraphModule, Node
from .passes import DeadCodeElimination, PassBase, PassResult


_DECOMP_METHODS: Dict[str, Callable[[Graph, Node], Node]] = {}


def _method(name: str):
    def register(fn: Callable[[Graph, Node], Node]) -> None:
        _DECOMP_METHODS[name] = fn

    return register


@_method("sigmoid")
def _sigmoid(graph: Graph, node: Node) -> Node:
    """sigmoid(x) -> 1 / (1 + exp(-x))"""
    x = node.args[0]
    neg_x = graph.create_node("call_function", operator.neg, (x,))
    exp_x = graph.create_node("call_method", "exp", (neg_x,))
    one = graph.create_node("call_function", operator.add, (exp_x, 1))
    return graph.create_node("call_function", operator.truediv, (1, one))


@_method("silu")
def _silu(graph: Graph, node: Node) -> Node:
    """silu(x) -> x * sigmoid(x)"""
    x = node.args[0]
    sig = _DECOMP_METHODS["sigmoid"](graph, node)
    return graph.create_node("call_function", operator.mul, (x, sig))


@_method("reciprocal")
def _reciprocal(graph: Graph, node: Node) -> Node:
    x = node.args[0]
    return graph.create_node("call_function", operator.truediv, (1, x))


@_method("square")
def _square(graph: Graph, node: Node) -> Node:
    x = node.args[0]
    return graph.create_node("call_function", operator.mul, (x, x))


class DecomposePass(PassBase):
    """Rewrite registered composite methods into derivative-covered primitives."""

    def __call__(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        changed = False
        for node in list(graph.nodes):
            if node.op != "call_method":
                continue
            rule = _DECOMP_METHODS.get(node.target)
            if rule is None:
                continue
            # Replacement sub-chains must precede the replaced node's users,
            # but create_node appends. Capture what the rule added and move
            # those nodes just before the original site.
            start = len(graph.nodes)
            replacement = rule(graph, node)
            created = graph.nodes[start:]
            if created:
                pos = graph.nodes.index(node)
                for offset, new_node in enumerate(created):
                    graph.nodes.remove(new_node)
                    graph.nodes.insert(pos + offset, new_node)
            node.replace_all_uses_with(replacement)
            changed = True
        if not changed:
            return PassResult(graph_module, False)
        DeadCodeElimination()(graph_module)
        return PassResult(graph_module, True)
