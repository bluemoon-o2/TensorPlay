"""Static fusion segmentation for the Stax backend (L5-M5c).

Splits a captured graph into an ordered list of fusion segments using the
Inductor vertical-fusion rule in miniature:

* pointwise runs merge into one segment;
* a pointwise run may end with ONE reduction epilogue (``x.sum(dim)``
  family) — the pw→red vertical fusion;
* a pure pointwise chain that transitively consumes the reduction result
  folds back INTO the same kernel as a store-time epilogue (red→pw
  vertical fusion, ``Segment.epilogue``); chains reading anything else
  start a new kernel;
* back-to-back reductions split into separate segments;
* any non-pointwise, non-reduction operator is an extern barrier and the
  whole graph falls back (v1; mixed interpreted/compiled stitching is a
  later milestone).

The scheduler owns no operator knowledge itself: the backend injects the
pointwise predicate and the reduction classifier, so this module stays free
of imports from ``backends``/``codegen`` (no cycles, one source of truth for
fusibility decisions — the old ad-hoc whole-graph detectors delegate here).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

from .graph import GraphModule, Node


@dataclass
class Segment:
    """One planned kernel region."""

    nodes: Tuple[Node, ...]
    #: "pw" (pure pointwise) or "pw+red" (pointwise prologue + reduction tail)
    kind: str
    #: classifier result for the reduction tail when ``kind == "pw+red"``
    reduction: Any = None
    #: pointwise chain computed on the reduction result INSIDE the same
    ## kernel (store-time epilogue, Inductor's red→pw vertical fusion).
    #: Every node here transitively consumes ``nodes[-1]``; anything else
    ## stays a separate segment.
    epilogue: Tuple[Node, ...] = ()

    @property
    def tail(self) -> Node | None:
        return self.nodes[-1] if self.nodes else None

    @property
    def export_node(self) -> Node | None:
        """Last value this segment produces (epilogue tail when present)."""

        if self.epilogue:
            return self.epilogue[-1]
        return self.tail

    @property
    def producer(self) -> Node | None:
        """Input of the reduction tail (the pointwise chain result)."""

        if self.kind == "pw+red" and self.tail is not None:
            value = self.tail.args[0]
            return value if isinstance(value, Node) else None
        return None


def _flatten_values(value: Any):
    """Yield every ``Node`` inside a (possibly nested) arg structure."""

    if isinstance(value, Node):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_values(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _flatten_values(item)


def _epilogue_join(
    node: Node,
    reduction_tail: Node,
    epilogue_run: Tuple[Node, ...],
) -> bool:
    """May ``node`` continue the in-kernel epilogue after a reduction?

    The node must depend on something already living in the kernel's
    register space — the reduction result or an earlier epilogue node —
    and may otherwise read only graph placeholders (or scalars).  A pure
    pointwise node over placeholders alone would need the pre-reduction
    tile, which no longer exists past the reduction.
    """

    live = {reduction_tail, *epilogue_run}
    touches_live = False
    for dep in set(_flatten_values(node.args)) | set(
        _flatten_values(node.kwargs)
    ):
        if dep in live:
            touches_live = True
        elif dep.op != "placeholder":
            return False
    return touches_live


def segment_graph(
    graph_module: GraphModule,
    *,
    is_pointwise: Callable[[Node], bool],
    classify_reduction: Callable[[Node], Any],
) -> Optional[List[Segment]]:
    """Partition ``graph_module`` into ordered fusion segments.

    Returns ``None`` when the graph mixes in operators this backend cannot
    lower yet (extern barrier → interpreted fallback for the whole region).
    """

    segments: List[Segment] = []
    current: List[Node] = []
    current_reduction: Any = None
    current_epilogue: List[Node] = []

    def close() -> None:
        nonlocal current, current_reduction, current_epilogue
        if not current:
            return
        kind = "pw+red" if current_reduction is not None else "pw"
        segments.append(
            Segment(
                nodes=tuple(current),
                kind=kind,
                reduction=current_reduction,
                epilogue=tuple(current_epilogue),
            )
        )
        current = []
        current_reduction = None
        current_epilogue = []

    for node in graph_module.graph.nodes:
        if node.op in {"placeholder", "output"}:
            continue
        reduction = classify_reduction(node)
        if reduction is not None:
            if current_reduction is not None:
                # reduction feeding a reduction: kernel boundary between them
                close()
                current = [node]
                current_reduction = reduction
                continue
            current.append(node)
            current_reduction = reduction
            continue
        if is_pointwise(node):
            if current_reduction is not None:
                # pw after a reduction joins the SAME kernel as a store-time
                # epilogue when it lives on the reduction's registers;
                # otherwise the kernel boundary falls here (v1 rule).
                if _epilogue_join(node, current[-1], tuple(current_epilogue)):
                    current_epilogue.append(node)
                    continue
                close()
            current.append(node)
            continue
        # extern operator: v1 falls back for the whole graph
        return None

    close()
    return segments or None


def describe(segments: List[Segment]) -> str:
    """Compact human-readable schedule, e.g. ``pw+red+ep -> pw``."""

    parts = []
    for segment in segments:
        label = segment.kind
        if segment.epilogue:
            label += "+ep"
        parts.append(label)
    return " -> ".join(parts)


def annotate(
    graph_module: GraphModule, segments: List[Segment]
) -> None:
    """Record the plan on the GraphModule for backends/debug tooling."""

    graph_module.meta["stax_segments"] = [
        {
            "kind": segment.kind,
            "nodes": [node.name for node in segment.nodes],
            "epilogue": [node.name for node in segment.epilogue],
            "reduction": getattr(segment.reduction, "__dict__", segment.reduction),
        }
        for segment in segments
    ]
