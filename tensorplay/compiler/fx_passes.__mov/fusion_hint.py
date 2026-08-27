"""Pointwise fusion-region annotation for the Stax/Triton code generators.

``PointwiseFusionHint`` walks the graph, identifies maximal chains of
pointwise ops (the ops Stax's fused lowering supports) and stamps each node
with ``meta["fusion_hint"] = "pointwise"`` plus a region id.  Code generators
consume the hints instead of re-deriving fusibility, and mixed graphs can
state exactly which subgraph is fusable.

The op set is the single source of truth shared with
``tensorplay/backends/stax.py`` (which imports it), replacing the previously
duplicated private constant.
"""

from __future__ import annotations

from ..passes import PassBase, PassResult

__all__ = ["POINTWISE_FUSED_OP_NAMES", "PointwiseFusionHint"]

# Keep in sync with stax's fused-lowering support; stax imports this set.
POINTWISE_FUSED_OP_NAMES = frozenset(
    {
        "add",
        "sub",
        "mul",
        "div",
        "pow",
        "neg",
        "pos",
        "abs",
        "sin",
        "cos",
        "exp",
        "log",
        "sigmoid",
        "sqrt",
        "square",
        "tanh",
        "relu",
    }
)


def _op_name(node):
    target = node.target
    if node.op == "call_function":
        return getattr(target, "__name__", str(target))
    if node.op == "call_method":
        return str(target)
    return None


class PointwiseFusionHint(PassBase):
    """Annotate maximal pointwise regions with fusion hints.

    A node belongs to a region when it is pointwise-fusible; producer edges
    between fusible nodes merge them into one region.  Non-fusible ops act
    as region boundaries.
    """

    def __call__(self, graph_module) -> PassResult:
        graph = graph_module.graph
        modified = False

        sample_node = next((n for n in graph.nodes if n.op != "output"), None)
        node_type = type(sample_node) if sample_node is not None else None

        def is_node(value) -> bool:
            return node_type is not None and isinstance(value, node_type)

        def fusible(node) -> bool:
            if id(node) in self._cache:
                return self._cache[id(node)]
            name = _op_name(node)
            ok = (
                name in POINTWISE_FUSED_OP_NAMES
                and not node.kwargs
                and len(node.args) <= 3
            )
            self._cache[id(node)] = ok
            return ok

        self._cache: dict[int, bool] = {}

        nodes = [n for n in graph.nodes if n.op not in {"placeholder", "output"}]
        parent: dict[int, int] = {id(n): id(n) for n in nodes if fusible(n)}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for node in nodes:
            if not fusible(node):
                continue
            for arg in node.args:
                if is_node(arg) and id(arg) in parent:
                    ra, rb = find(id(node)), find(id(arg))
                    if ra != rb:
                        parent[rb] = ra

        region_of: dict[int, int] = {}
        for key in parent:
            root = find(key)
            region_of.setdefault(root, len(region_of))

        for node in nodes:
            if not fusible(node):
                continue
            region = region_of[find(id(node))]
            if (
                node.meta.get("fusion_hint") != "pointwise"
                or node.meta.get("fusion_region") != region
            ):
                modified = True
            node.meta["fusion_hint"] = "pointwise"
            node.meta["fusion_region"] = region

        return PassResult(graph_module, modified)
