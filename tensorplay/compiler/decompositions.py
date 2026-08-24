"""Operator decomposition pass (L5-M4).

Modeled on ``torch/_inductor/decomposition.py``: rewrite composite
operators into the primitive set *before* AOT, so the derivative registry
only has to cover primitives. Every expansion below uses ops that already
have local vector-Jacobian rules, which keeps decomposed graphs
differentiable by construction.

This table deliberately lands every rewrite on the Stax/Triton fusible
primitive set (``POINTWISE_FUSED_OP_NAMES``): add/sub/mul/div/pow/neg/exp/
log/sqrt/tanh/relu/sigmoid/square plus scalar constants.  That is the same
lever Inductor pulls — a small kernel set covers a wide operator surface —
so each entry below multiplies *natively compiled* coverage without any
new kernel.  Compare-based rewrites (elu/selu/hardshrink/sign family,
which need ``where`` + comparison primitives) are intentionally deferred
until ``where``/``gt`` join the native set.

Rules are keyed by semantic op name and fire on both ``call_method`` and
``call_function`` sites, matching how the tracer records tensor methods
versus ``tensorplay.nn.functional`` wrappers.
"""

from __future__ import annotations

import math
import operator
from typing import Any, Callable, Dict

from .graph import Graph, GraphModule, Node
from .passes import DeadCodeElimination, PassBase, PassResult


_DECOMP_METHODS: Dict[str, Callable[[Graph, Node], Node]] = {}


def _method(name: str):
    def register(fn: Callable[[Graph, Node], Node]) -> None:
        _DECOMP_METHODS[name] = fn

    return register


# --- small builders -------------------------------------------------------


def _unop(graph: Graph, kind: str, x: Any) -> Node:
    """Apply a tensor method (exp/log/sqrt/tanh/...) to ``x``."""

    return graph.create_node("call_method", kind, (x,))


def _binop(graph: Graph, op: Any, lhs: Any, rhs: Any) -> Node:
    return graph.create_node("call_function", op, (lhs, rhs))


# --- existing core rules ----------------------------------------------------


@_method("sigmoid")
def _sigmoid(graph: Graph, node: Node) -> Node:
    """sigmoid(x) -> 1 / (1 + exp(-x))"""
    x = node.args[0]
    neg_x = graph.create_node("call_function", operator.neg, (x,))
    exp_x = _unop(graph, "exp", neg_x)
    one = _binop(graph, operator.add, exp_x, 1)
    return _binop(graph, operator.truediv, 1, one)


@_method("silu")
def _silu(graph: Graph, node: Node) -> Node:
    """silu(x) -> x * sigmoid(x)"""
    x = node.args[0]
    sig = _DECOMP_METHODS["sigmoid"](graph, node)
    return _binop(graph, operator.mul, x, sig)


# swish is the historical alias of silu (torch parity).
_DECOMP_METHODS["swish"] = _silu


@_method("reciprocal")
def _reciprocal(graph: Graph, node: Node) -> Node:
    x = node.args[0]
    return _binop(graph, operator.truediv, 1, x)


@_method("square")
def _square(graph: Graph, node: Node) -> Node:
    x = node.args[0]
    return _binop(graph, operator.mul, x, x)


# --- activations ------------------------------------------------------------


@_method("softplus")
def _softplus(graph: Graph, node: Node) -> Node:
    """softplus(x) -> log(1 + exp(x))

    torch/_inductor lowers with log1p(exp(x)); log1p itself decomposes to
    log(1 + .) below, so both spellings converge on the same primitive chain.
    """
    x = node.args[0]
    exp_x = _unop(graph, "exp", x)
    summed = _binop(graph, operator.add, exp_x, 1)
    return _unop(graph, "log", summed)


@_method("mish")
def _mish(graph: Graph, node: Node) -> Node:
    """mish(x) -> x * tanh(softplus(x))"""
    x = node.args[0]
    softplus = _DECOMP_METHODS["softplus"](graph, node)
    tanh_sp = _unop(graph, "tanh", softplus)
    return _binop(graph, operator.mul, x, tanh_sp)


@_method("tanhshrink")
def _tanhshrink(graph: Graph, node: Node) -> Node:
    """tanhshrink(x) -> x - tanh(x)"""
    x = node.args[0]
    tanh_x = _unop(graph, "tanh", x)
    return _binop(graph, operator.sub, x, tanh_x)


# --- logarithm / exponential family -----------------------------------------


@_method("log1p")
def _log1p(graph: Graph, node: Node) -> Node:
    x = node.args[0]
    summed = _binop(graph, operator.add, x, 1)
    return _unop(graph, "log", summed)


@_method("expm1")
def _expm1(graph: Graph, node: Node) -> Node:
    x = node.args[0]
    exp_x = _unop(graph, "exp", x)
    return _binop(graph, operator.sub, exp_x, 1)


@_method("exp2")
def _exp2(graph: Graph, node: Node) -> Node:
    """exp2(x) -> 2 ** x"""
    x = node.args[0]
    return _binop(graph, operator.pow, 2, x)


@_method("log10")
def _log10(graph: Graph, node: Node) -> Node:
    x = node.args[0]
    log_x = _unop(graph, "log", x)
    return _binop(
        graph, operator.mul, log_x, 1.0 / math.log(10.0)
    )


@_method("logit")
def _logit(graph: Graph, node: Node) -> Node:
    """logit(x) -> log(x) - log(1 - x)

    torch/_inductor writes this as log(x / (1 - x)); the subtract spelling
    avoids a division whose numerator/denominator can both underflow.
    """
    x = node.args[0]
    log_x = _unop(graph, "log", x)
    one_minus = _binop(graph, operator.sub, 1, x)
    log_one_minus = _unop(graph, "log", one_minus)
    return _binop(graph, operator.sub, log_x, log_one_minus)


# --- hyperbolic / trig family ------------------------------------------------


@_method("sinh")
def _sinh(graph: Graph, node: Node) -> Node:
    """sinh(x) -> (e - 1/e) / 2"""
    x = node.args[0]
    exp_x = _unop(graph, "exp", x)
    inv_exp = _binop(graph, operator.truediv, 1, exp_x)
    diff = _binop(graph, operator.sub, exp_x, inv_exp)
    return _binop(graph, operator.mul, diff, 0.5)


@_method("cosh")
def _cosh(graph: Graph, node: Node) -> Node:
    """cosh(x) -> (e + 1/e) / 2"""
    x = node.args[0]
    exp_x = _unop(graph, "exp", x)
    inv_exp = _binop(graph, operator.truediv, 1, exp_x)
    total = _binop(graph, operator.add, exp_x, inv_exp)
    return _binop(graph, operator.mul, total, 0.5)


@_method("asinh")
def _asinh(graph: Graph, node: Node) -> Node:
    x = node.args[0]
    sq = _binop(graph, operator.mul, x, x)
    inner = _binop(graph, operator.add, sq, 1)
    root = _unop(graph, "sqrt", inner)
    total = _binop(graph, operator.add, x, root)
    return _unop(graph, "log", total)


@_method("acosh")
def _acosh(graph: Graph, node: Node) -> Node:
    x = node.args[0]
    sq = _binop(graph, operator.mul, x, x)
    inner = _binop(graph, operator.sub, sq, 1)
    root = _unop(graph, "sqrt", inner)
    total = _binop(graph, operator.add, x, root)
    return _unop(graph, "log", total)


@_method("atanh")
def _atanh(graph: Graph, node: Node) -> Node:
    """atanh(x) -> log((1 + x) / (1 - x)) / 2"""
    x = node.args[0]
    num = _binop(graph, operator.add, 1, x)
    den = _binop(graph, operator.sub, 1, x)
    ratio = _binop(graph, operator.truediv, num, den)
    logged = _unop(graph, "log", ratio)
    return _binop(graph, operator.mul, logged, 0.5)


@_method("sec")
def _sec(graph: Graph, node: Node) -> Node:
    x = node.args[0]
    cos_x = _unop(graph, "cos", x)
    return _binop(graph, operator.truediv, 1, cos_x)


@_method("csc")
def _csc(graph: Graph, node: Node) -> Node:
    x = node.args[0]
    sin_x = _unop(graph, "sin", x)
    return _binop(graph, operator.truediv, 1, sin_x)


@_method("cot")
def _cot(graph: Graph, node: Node) -> Node:
    x = node.args[0]
    cos_x = _unop(graph, "cos", x)
    sin_x = _unop(graph, "sin", x)
    return _binop(graph, operator.truediv, cos_x, sin_x)


# --- angle / scaling conversions ---------------------------------------------


@_method("rad2deg")
def _rad2deg(graph: Graph, node: Node) -> Node:
    x = node.args[0]
    return _binop(graph, operator.mul, x, 180.0 / math.pi)


@_method("deg2rad")
def _deg2rad(graph: Graph, node: Node) -> Node:
    x = node.args[0]
    return _binop(graph, operator.mul, x, math.pi / 180.0)


# --- binary composites --------------------------------------------------------


@_method("squared_difference")
def _squared_difference(graph: Graph, node: Node) -> Node:
    lhs, rhs = node.args[0], node.args[1]
    diff = _binop(graph, operator.sub, lhs, rhs)
    return _binop(graph, operator.mul, diff, diff)


@_method("lerp")
def _lerp(graph: Graph, node: Node) -> Node:
    """lerp(start, end, weight) -> start + weight * (end - start)

    Scalar-weight and tensor-weight spellings both work: ``weight`` flows
    through as an operand and broadcasts like any primitive input.
    """

    args = tuple(node.args) + (
        node.kwargs.get("weight") if node.kwargs else None,
    )
    start, end = args[0], args[1]
    weight = args[2] if len(args) > 2 else None
    if weight is None:
        raise ValueError("lerp decomposition requires a weight argument")
    diff = _binop(graph, operator.sub, end, start)
    scaled = _binop(graph, operator.mul, weight, diff)
    return _binop(graph, operator.add, start, scaled)


@_method("addcmul")
def _addcmul(graph: Graph, node: Node) -> Node:
    """addcmul(input, t1, t2, value=1) -> input + value * t1 * t2"""

    value = node.kwargs.get("value", 1) if node.kwargs else 1
    inp, t1, t2 = node.args[0], node.args[1], node.args[2]
    product = _binop(graph, operator.mul, t1, t2)
    scaled = (
        product
        if value == 1
        else _binop(graph, operator.mul, product, value)
    )
    return _binop(graph, operator.add, inp, scaled)


@_method("addcdiv")
def _addcdiv(graph: Graph, node: Node) -> Node:
    """addcdiv(input, t1, t2, value=1) -> input + value * t1 / t2"""

    value = node.kwargs.get("value", 1) if node.kwargs else 1
    inp, t1, t2 = node.args[0], node.args[1], node.args[2]
    quotient = _binop(graph, operator.truediv, t1, t2)
    scaled = (
        quotient
        if value == 1
        else _binop(graph, operator.mul, quotient, value)
    )
    return _binop(graph, operator.add, inp, scaled)


class DecomposePass(PassBase):
    """Rewrite registered composite methods into derivative-covered primitives."""

    def __call__(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        changed = False
        for node in list(graph.nodes):
            if node.op == "call_method":
                name = node.target if isinstance(node.target, str) else None
            elif node.op == "call_function":
                name = getattr(node.target, "__name__", None)
            else:
                continue
            if name is None:
                continue
            rule = _DECOMP_METHODS.get(name)
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
