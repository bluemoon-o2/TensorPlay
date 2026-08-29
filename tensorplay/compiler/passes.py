"""Pass management utilities.

A pass receives the :class:`~tensorplay.compiler.graph.GraphModule`, mutates
its graph in place, and reports whether it changed anything.  The
:class:`PassManager` runs passes in order to a fixpoint, linting after every
round so a broken transform fails loudly instead of poisoning backends.
"""

from __future__ import annotations

import operator
from typing import Any, Callable, List, NamedTuple, Optional, Sequence, Tuple

from .graph import (
    Graph,
    GraphCaptureError,
    GraphModule,
    Node,
    Proxy,
    _iter_nodes,
    _map_arg,
    dead_code_elimination,
)

__all__ = [
    "ConstFold",
    "DeadCodeElimination",
    "PassBase",
    "PassManager",
    "PassResult",
    "ShapeProp",
]


class PassResult(NamedTuple):
    graph_module: GraphModule
    modified: bool


class PassBase:

    def __call__(self, graph_module: GraphModule) -> PassResult:
        raise NotImplementedError

    def constraint(self) -> Optional[str]:
        """Human-readable precondition; ``None`` means unconditional."""
        return None


def _as_graph_module(target: Any) -> GraphModule:
    if isinstance(target, GraphModule):
        return target
    if isinstance(target, Graph):
        return GraphModule(None, target, None)
    raise TypeError(
        f"passes expect a GraphModule or Graph, got {type(target)!r}"
    )


class PassManager:
    """Run passes until no pass reports a modification (or once).

    pass once and validates the graph; iteration stops when a full round is
    unmodified or ``max_iterations`` is reached.
    """

    def __init__(
        self,
        passes: Sequence[Callable[[GraphModule], Any]] = (),
        *,
        run_passes_once: bool = False,
        max_iterations: int = 16,
    ) -> None:
        self.passes = list(passes)
        self.run_passes_once = run_passes_once
        self.max_iterations = max_iterations

    def add(self, fn: Callable[[GraphModule], Any]) -> "PassManager":
        self.passes.append(fn)
        return self

    def __call__(self, target: Any) -> PassResult:
        graph_module = _as_graph_module(target)
        any_modified = False
        rounds = 1 if self.run_passes_once else self.max_iterations
        for _ in range(rounds):
            round_modified = False
            for fn in self.passes:
                result = fn(graph_module)
                modified = (
                    bool(result.modified)
                    if isinstance(result, PassResult)
                    else bool(result)
                )
                round_modified = round_modified or modified
            graph_module.graph.lint()
            if not round_modified:
                break
            any_modified = True
        return PassResult(graph_module, any_modified)


# ---------------------------------------------------------------------------
# Built-in passes
# ---------------------------------------------------------------------------


class DeadCodeElimination(PassBase):
    """Drop nodes that cannot reach the output (placeholders are kept)."""

    def __call__(self, graph_module: GraphModule) -> PassResult:
        removed = dead_code_elimination(graph_module.graph)
        return PassResult(graph_module, removed > 0)


_FOLDABLE_TARGETS = frozenset(
    {
        operator.add,
        operator.sub,
        operator.mul,
        operator.truediv,
        operator.floordiv,
        operator.mod,
        operator.pow,
        operator.neg,
        operator.pos,
        operator.abs,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
        operator.eq,
        operator.ne,
        operator.getitem,
    }
)


def _is_tensor_like(value: Any) -> bool:
    shape = getattr(value, "shape", None)
    return shape is not None and hasattr(value, "dtype")


def _replace_with_constant(node: Node, value: Any) -> None:
    """Rewrite every use of ``node`` into the literal ``value``, then erase.

    Constant replacement cannot reuse :meth:`Node.replace_all_uses_with`
    because the folded result has no node of its own.
    """

    for user in list(node.users):
        user.args = _map_arg(user.args, lambda v: value if v is node else v)
        user.kwargs = _map_arg(user.kwargs, lambda v: value if v is node else v)
        node.users.discard(user)
    node.erase_node()


class ConstFold(PassBase):
    """Evaluate pure scalar subgraphs at compile time.

    Only whitelisted ``operator`` targets on arguments free of nodes are
    folded; tensor-like operands keep their runtime semantics untouched.
    """

    def __call__(self, graph_module: GraphModule) -> PassResult:
        modified = False
        for node in list(graph_module.graph.nodes):
            if node.op != "call_function" or node.target not in _FOLDABLE_TARGETS:
                continue
            if next(_iter_nodes(node.args), None) is not None:
                continue
            if next(_iter_nodes(node.kwargs), None) is not None:
                continue
            args = tuple(node.args)
            if any(_is_tensor_like(item) for item in _iter_flat(args)):
                continue
            try:
                value = node.target(*args, **node.kwargs)
            except Exception:
                # Runtime-invalid constants (e.g. division by zero) must keep
                # failing at execution time, not silently change behavior.
                continue
            _replace_with_constant(node, value)
            modified = True
        return PassResult(graph_module, modified)


def _iter_flat(values: Tuple[Any, ...]):
    for item in values:
        yield item


class ShapeProp(PassBase):
    """Execute the graph on example inputs, recording values into node meta.

    After running, every node carries ``meta["val"]`` and tensor-valued nodes
    additionally carry ``meta["tensor_shape"]`` (consumed by
    :meth:`Graph.to_dot` tooltips and downstream shape-aware passes).  Meta
    enrichment is not a structural modification, so this pass always reports
    ``modified=False`` and never loops the manager.
    """

    def __init__(self, example_inputs: Sequence[Any]) -> None:
        self.example_inputs = tuple(example_inputs)

    def __call__(self, graph_module: GraphModule) -> PassResult:
        # Interpreting invokes call_function targets directly; symbolic
        # targets (Node/Proxy) would append fresh nodes to the finished
        # graph, so such regions are left unannotated instead of corrupted.
        for node in graph_module.graph.nodes:
            if node.op == "call_function" and isinstance(node.target, (Node, Proxy)):
                return PassResult(graph_module, False)
        # Bind by placeholder name: backend inputs follow placeholder order,
        # which is not necessarily valid positional order for signatures with
        # keyword-only parameters.
        bindings = {
            node.name: value
            for node, value in zip(
                graph_module.graph.placeholders, self.example_inputs
            )
        }
        graph_module._interpret(_record_meta=True, **bindings)
        return PassResult(graph_module, False)
