"""The canonical TensorPlay compiler graph.

This module is deliberately independent of Stax.  A compiler frontend owns
capture and produces this graph; backends consume :class:`GraphModule`.
Keeping the graph here mirrors the PyTorch split between Dynamo/FX and a
backend such as Inductor.
"""

from __future__ import annotations

import inspect
import keyword
import operator
import re
import types
from typing import Any, Callable, Dict, Iterable, Optional, Tuple


class GraphCaptureError(RuntimeError):
    """Raised when Python code cannot be represented by the current graph."""


def _map_arg(value: Any, fn: Callable[[Any], Any]) -> Any:
    if isinstance(value, (Node, Proxy)):
        return fn(value)
    if isinstance(value, tuple):
        return tuple(_map_arg(item, fn) for item in value)
    if isinstance(value, list):
        return [_map_arg(item, fn) for item in value]
    if isinstance(value, dict):
        return {key: _map_arg(item, fn) for key, item in value.items()}
    if isinstance(value, slice):
        return slice(
            _map_arg(value.start, fn),
            _map_arg(value.stop, fn),
            _map_arg(value.step, fn),
        )
    return value


def _iter_proxies(value: Any) -> Iterable["Proxy"]:
    if isinstance(value, Proxy):
        yield value
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_proxies(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_proxies(item)
        return
    if isinstance(value, slice):
        yield from _iter_proxies(value.start)
        yield from _iter_proxies(value.stop)
        yield from _iter_proxies(value.step)


def capture_call(
    target: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Optional["Proxy"]:
    """Capture a Python functional operator when any argument is symbolic.

    TensorPlay's generated functional wrappers call into the native extension
    directly, so the extension cannot see a :class:`Proxy`.  This small
    dispatcher is the equivalent of the operator-overload dispatch that lets
    FX/Dynamo record ``torch.nn.functional`` calls without changing their
    eager implementation.
    """

    proxies = list(_iter_proxies(args))
    proxies.extend(_iter_proxies(kwargs))
    if not proxies:
        return None
    tracer = proxies[0].tracer
    if any(proxy.tracer is not tracer for proxy in proxies[1:]):
        raise GraphCaptureError("cannot combine proxies from different traces")
    return tracer.create_proxy("call_function", target, args, kwargs)


def _iter_nodes(value: Any) -> Iterable["Node"]:
    if isinstance(value, Node):
        yield value
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_nodes(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_nodes(item)
        return
    if isinstance(value, slice):
        yield from _iter_nodes(value.start)
        yield from _iter_nodes(value.stop)
        yield from _iter_nodes(value.step)


def _snake_case(name: str) -> str:
    """Port of torch.fx.Graph._snake_case for semantic node naming."""

    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _sanitize_name(name: str) -> str:
    """Reduce a candidate name to a valid Python identifier."""

    sanitized = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    if not sanitized or sanitized[0].isdigit() or keyword.iskeyword(sanitized):
        sanitized = f"_{sanitized}"
    return sanitized


def _target_to_str(target: Any) -> str:
    """Derive a readable base name from a node target (fx-style)."""

    if isinstance(target, str):
        return _snake_case(target.split(".")[-1])
    if callable(target):
        atom = getattr(target, "__name__", None) or type(target).__name__
        return _snake_case(str(atom))
    return type(target).__name__


def _format_target(target: Any) -> str:
    """Human-readable rendering of a node target for visualization."""

    name = getattr(target, "__name__", None)
    if isinstance(target, str):
        return target
    if callable(target) and name:
        module = getattr(target, "__module__", "") or ""
        if module and module != "builtins":
            return f"{module}.{name}"
        return str(name)
    if name:
        return str(name)
    return repr(target)


class Node:
    """A single operation in the canonical compiler graph."""

    __slots__ = (
        "graph",
        "name",
        "op",
        "target",
        "args",
        "kwargs",
        "users",
        "meta",
    )

    def __init__(
        self,
        graph: "Graph",
        name: str,
        op: str,
        target: Any,
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.graph = graph
        self.name = name
        self.op = op
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}
        self.users: set[Node] = set()
        self.meta: Dict[str, Any] = {}

    def __repr__(self) -> str:
        return f"{self.name} = {self.op}[{self.target!r}]"

    def replace_all_uses_with(self, replace_with: "Node") -> int:
        """Rewrite every consumer of this node to consume ``replace_with``.

        Returns the number of uses rewritten.  The replacement must already
        be topologically ordered before this node, otherwise the graph would
        become invalid.
        """

        if replace_with is self:
            raise GraphCaptureError("cannot replace uses of a node with itself")
        positions = {node: index for index, node in enumerate(self.graph.nodes)}
        users = list(self.users)
        for user in users:
            if positions[replace_with] > positions[user]:
                raise GraphCaptureError(
                    f"cannot use {replace_with.name} to replace {self.name} "
                    f"in {user.name}: it appears later in the graph"
                )

        def substitute(value: Any) -> Any:
            return replace_with if value is self else value

        for user in users:
            user.args = _map_arg(user.args, substitute)
            user.kwargs = _map_arg(user.kwargs, substitute)
            self.users.discard(user)
            replace_with.users.add(user)
        return len(users)

    def erase_node(self) -> None:
        """Remove this node from its graph.

        The node must have no remaining users; call
        :meth:`replace_all_uses_with` or dead code elimination first.
        Its name becomes reusable once no live node holds it.
        """

        if self.graph is None:
            raise GraphCaptureError(f"{self.name} has already been erased")
        if self.users:
            raise GraphCaptureError(
                f"cannot erase {self.name} because it still has "
                f"{len(self.users)} user(s); run dead code elimination first"
            )
        for input_node in (*_iter_nodes(self.args), *_iter_nodes(self.kwargs)):
            input_node.users.discard(self)
        self.graph.nodes.remove(self)
        self.graph = None


class Graph:
    """A mutable, topologically ordered operation graph."""

    def __init__(self) -> None:
        self.nodes: list[Node] = []

    @property
    def placeholders(self) -> list[Node]:
        return [node for node in self.nodes if node.op == "placeholder"]

    @property
    def outputs(self) -> list[Node]:
        return [node for node in self.nodes if node.op == "output"]

    @property
    def output_node(self) -> Node:
        """The single output node of this graph.

        :meth:`output` enforces the single-output invariant by replacing any
        previous output node, mirroring torch.fx.
        """

        outputs = self.outputs
        if not outputs:
            raise GraphCaptureError("graph has no output node")
        return outputs[0]

    def _create_unique_name(self, candidate: str) -> str:
        """Uniquify ``candidate`` against the currently live nodes.

        Names of erased nodes may be reused safely: uniqueness only matters
        between live nodes (generated executors emit one variable per node).
        """

        base = _sanitize_name(candidate)
        taken = {node.name for node in self.nodes}
        name = base
        suffix = 0
        while name in taken:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    def create_node(
        self,
        op: str,
        target: Any,
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None,
        *,
        name: Optional[str] = None,
    ) -> Node:
        def normalize(value: Any) -> Any:
            if isinstance(value, Proxy):
                return value.node
            return value

        normalized_args = _map_arg(args, normalize)
        normalized_kwargs = _map_arg(kwargs or {}, normalize)
        node_name = self._create_unique_name(
            name if name is not None else _target_to_str(target)
        )
        node = Node(
            self,
            node_name,
            op,
            target,
            tuple(normalized_args),
            dict(normalized_kwargs),
        )
        self.nodes.append(node)
        for input_node in _iter_nodes(node.args):
            input_node.users.add(node)
        for input_node in _iter_nodes(node.kwargs):
            input_node.users.add(node)
        return node

    def placeholder(self, name: str, default: Any = inspect.Parameter.empty) -> Node:
        node = self.create_node("placeholder", name, name=name)
        if default is not inspect.Parameter.empty:
            node.meta["default"] = default
        return node

    def get_attr(self, qualified_name: str) -> Node:
        return self.create_node("get_attr", qualified_name)

    def call_module(self, qualified_name: str, args: Tuple[Any, ...], kwargs: Optional[Dict[str, Any]] = None) -> Node:
        return self.create_node("call_module", qualified_name, args, kwargs)

    def call_function(self, target: Callable[..., Any], args: Tuple[Any, ...], kwargs: Optional[Dict[str, Any]] = None) -> Node:
        return self.create_node("call_function", target, args, kwargs)

    def call_method(self, method_name: str, args: Tuple[Any, ...], kwargs: Optional[Dict[str, Any]] = None) -> Node:
        return self.create_node("call_method", method_name, args, kwargs)

    def output(self, value: Any) -> Node:
        """Declare the graph result.

        The graph keeps exactly one output node: creating a new one replaces
        the previous output instead of appending a second node.  This keeps
        the interpreter and the generated executor in agreement about which
        value the graph returns.
        """

        for old_output in list(self.outputs):
            old_output.erase_node()
        return self.create_node("output", "output", (value,), name="output")

    def lint(self) -> None:
        positions = {node: index for index, node in enumerate(self.nodes)}
        for node in self.nodes:
            for input_node in (*_iter_nodes(node.args), *_iter_nodes(node.kwargs)):
                if positions[input_node] >= positions[node]:
                    raise GraphCaptureError(
                        f"Graph is not topologically ordered: {input_node} -> {node}"
                    )
                if input_node.graph is not self:
                    raise GraphCaptureError(
                        f"Node {node.name} references {input_node.name} "
                        "from another graph"
                    )
        if len(self.outputs) > 1:
            raise GraphCaptureError(
                f"Graph must have a single output node; found {len(self.outputs)}"
            )

    def eliminate_dead_code(self) -> bool:
        """Remove pure nodes that cannot reach the graph output."""

        return dead_code_elimination(self)

    def to_dot(
        self,
        *,
        graph_name: str = "TensorPlayGraph",
        rankdir: str = "TB",
        show_shapes: bool = True,
    ) -> str:
        """Render this graph as a Graphviz DOT source string.

        The output is plain text and requires no third-party packages; feed it
        to ``dot -Tpng`` or :meth:`draw` to produce an image.  Styling follows
        the conventions popularized by torchviz/torchview: inputs are blue
        ellipses, operations yellow boxes, submodule calls green components,
        attributes orange diamonds, and the result a pale-green ellipse.
        """

        styles: Dict[str, Tuple[str, str]] = {
            "placeholder": ("ellipse", "#aec7e8"),
            "get_attr": ("diamond", "#ffbb78"),
            "call_function": ("box", "#ffffb3"),
            "call_method": ("box", "#ffffb3"),
            "call_module": ("component", "#98df8a"),
            "output": ("ellipse", "#c7e9c0"),
        }

        def esc(text: Any) -> str:
            return (
                str(text)
                .replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("\n", "\\n")
            )

        lines = [
            "digraph {graph_name} {{".format(graph_name=esc(graph_name)),
            f"    rankdir={rankdir};",
            '    graph [fontname="monospace"];',
            '    node [fontname="monospace" fontsize=10 style=filled];',
            '    edge [fontsize=9];',
        ]
        for node in self.nodes:
            shape, color = styles.get(node.op, ("box", "#d9d9d9"))
            attrs = [f"label=\"{esc(node.name)}\\n{esc(node.op)}[{esc(_format_target(node.target))}]\""]
            if show_shapes and node.meta.get("tensor_shape") is not None:
                attrs.append(f"tooltip=\"{esc(node.meta['tensor_shape'])}\"")
            lines.append(
                f'    "{esc(node.name)}" [shape={shape}, fillcolor="{color}", {", ".join(attrs)}];'
            )
        for node in self.nodes:
            for input_node in _iter_nodes(node.args):
                lines.append(f'    "{esc(input_node.name)}" -> "{esc(node.name)}";')
            for key, value in node.kwargs.items():
                for input_node in _iter_nodes(value):
                    lines.append(
                        f'    "{esc(input_node.name)}" -> "{esc(node.name)}" '
                        f'[label="{esc(key)}"];'
                    )
        lines.append("}")
        return "\n".join(lines)

    def draw(
        self,
        filename: str,
        format: Optional[str] = None,
        *,
        rankdir: str = "TB",
    ) -> str:
        """Export this graph as an image (PNG/SVG/PDF) via Graphviz.

        Writes the DOT source next to ``filename`` and, when the ``dot``
        binary from Graphviz is available, renders it into an image file.
        Without Graphviz installed only the ``.gv`` source is produced so the
        caller can render it elsewhere (for example on
        https://dreampuf.github.io/GraphvizOnline).

        Args:
            filename: Output image path, e.g. ``"model.png"``.
            format: Image format inferred from the filename suffix by
                default (``png``, ``svg``, ``pdf``, ...).

        Returns:
            Path of the rendered image, or of the emitted ``.gv`` source when
            Graphviz is unavailable.
        """

        import shutil
        import subprocess
        import os
        from pathlib import Path

        stem = Path(filename).with_suffix("")
        fmt = format or Path(filename).suffix.lstrip(".") or "png"
        gv_path = Path(str(stem) + ".gv")
        gv_path.write_text(self.to_dot(rankdir=rankdir))

        dot_binary = shutil.which("dot")
        if dot_binary is None:
            raise RuntimeError(
                f"Graphviz executable 'dot' not found; wrote DOT source to "
                f"{gv_path}. Install graphviz (https://graphviz.org/download/) "
                f"and rerun draw(), or render {gv_path} with any DOT viewer."
            )
        output_path = f"{stem}.{fmt}"
        subprocess.run(
            [dot_binary, "-T" + fmt, str(gv_path), "-o", output_path],
            check=True,
        )
        return output_path

    def python_code(self) -> str:
        lines = ["def forward(*args, **kwargs):"]
        for node in self.nodes:
            if node.op in {"placeholder", "output"}:
                continue
            lines.append(f"    # {node}")
        lines.append("    ...")
        return "\n".join(lines)


class Proxy:
    """Symbolic value used while the frontend captures Python operations."""

    __slots__ = ("node", "tracer")

    def __init__(self, node: Node, tracer: "Tracer") -> None:
        self.node = node
        self.tracer = tracer

    def _binary(self, target: Any, other: Any) -> "Proxy":
        return self.tracer.create_proxy("call_function", target, (self, other), {})

    def _unary(self, target: Any) -> "Proxy":
        return self.tracer.create_proxy("call_function", target, (self,), {})

    def __add__(self, other: Any) -> "Proxy":
        return self._binary(operator.add, other)

    def __radd__(self, other: Any) -> "Proxy":
        return self.tracer.create_proxy("call_function", operator.add, (other, self), {})

    def __sub__(self, other: Any) -> "Proxy":
        return self._binary(operator.sub, other)

    def __rsub__(self, other: Any) -> "Proxy":
        return self.tracer.create_proxy("call_function", operator.sub, (other, self), {})

    def __mul__(self, other: Any) -> "Proxy":
        return self._binary(operator.mul, other)

    def __rmul__(self, other: Any) -> "Proxy":
        return self.tracer.create_proxy("call_function", operator.mul, (other, self), {})

    def __truediv__(self, other: Any) -> "Proxy":
        return self._binary(operator.truediv, other)

    def __rtruediv__(self, other: Any) -> "Proxy":
        return self.tracer.create_proxy("call_function", operator.truediv, (other, self), {})

    def __floordiv__(self, other: Any) -> "Proxy":
        return self._binary(operator.floordiv, other)

    def __rfloordiv__(self, other: Any) -> "Proxy":
        return self.tracer.create_proxy("call_function", operator.floordiv, (other, self), {})

    def __mod__(self, other: Any) -> "Proxy":
        return self._binary(operator.mod, other)

    def __rmod__(self, other: Any) -> "Proxy":
        return self.tracer.create_proxy("call_function", operator.mod, (other, self), {})

    def __pow__(self, other: Any) -> "Proxy":
        return self._binary(operator.pow, other)

    def __rpow__(self, other: Any) -> "Proxy":
        return self.tracer.create_proxy("call_function", operator.pow, (other, self), {})

    def __matmul__(self, other: Any) -> "Proxy":
        return self._binary(operator.matmul, other)

    def __rmatmul__(self, other: Any) -> "Proxy":
        return self.tracer.create_proxy("call_function", operator.matmul, (other, self), {})

    def __lt__(self, other: Any) -> "Proxy":
        return self._binary(operator.lt, other)

    def __le__(self, other: Any) -> "Proxy":
        return self._binary(operator.le, other)

    def __eq__(self, other: Any) -> "Proxy":  # type: ignore[override]
        return self._binary(operator.eq, other)

    def __ne__(self, other: Any) -> "Proxy":  # type: ignore[override]
        return self._binary(operator.ne, other)

    def __gt__(self, other: Any) -> "Proxy":
        return self._binary(operator.gt, other)

    def __ge__(self, other: Any) -> "Proxy":
        return self._binary(operator.ge, other)

    def __neg__(self) -> "Proxy":
        return self._unary(operator.neg)

    def __pos__(self) -> "Proxy":
        return self._unary(operator.pos)

    def __getitem__(self, key: Any) -> "Proxy":
        return self.tracer.create_proxy("call_function", operator.getitem, (self, key), {})

    def _sample(self) -> Any:
        """Example value bound to this placeholder, if the tracer got one."""

        return self.tracer._samples.get(self.node.name)

    def _property(self, name: str) -> Any:
        """Resolve tensor metadata: concretely when a sample is available.

        Metadata (shape/dtype/device/...) is part of the compile signature,
        so specializing on it adds no new recompile conditions; data reads
        stay symbolic or raise.
        """

        sample = self._sample()
        self.tracer.metadata_touches.add((self.node.name, name))
        if sample is not None:
            return getattr(sample, name)
        return self.tracer.create_proxy("call_function", getattr, (self, name), {})

    @property
    def shape(self) -> Any:
        return self._property("shape")

    @property
    def dtype(self) -> "Proxy":
        return self._property("dtype")

    @property
    def device(self) -> "Proxy":
        return self._property("device")

    @property
    def ndim(self) -> "Proxy":
        return self._property("ndim")

    @property
    def requires_grad(self) -> "Proxy":
        return self._property("requires_grad")

    def __getattr__(self, name: str) -> Callable[..., "Proxy"]:
        def method(*args: Any, **kwargs: Any) -> "Proxy":
            return self.tracer.create_proxy(
                "call_method", name, (self, *args), kwargs
            )

        return method

    def __call__(self, *args: Any, **kwargs: Any) -> "Proxy":
        return self.tracer.create_proxy("call_function", self, args, kwargs)

    def sin(self) -> "Proxy":
        return self.tracer.create_proxy("call_method", "sin", (self,), {})

    def cos(self) -> "Proxy":
        return self.tracer.create_proxy("call_method", "cos", (self,), {})

    def exp(self) -> "Proxy":
        return self.tracer.create_proxy("call_method", "exp", (self,), {})

    def sqrt(self) -> "Proxy":
        return self.tracer.create_proxy("call_method", "sqrt", (self,), {})

    def relu(self) -> "Proxy":
        return self.tracer.create_proxy("call_method", "relu", (self,), {})

    def __bool__(self) -> bool:
        raise GraphCaptureError(
            "TensorPlay compiler cannot specialize a Proxy in Python control flow"
        )

    def __len__(self) -> int:
        sample = self._sample()
        self.tracer.metadata_touches.add((self.node.name, "len"))
        if sample is not None:
            if hasattr(sample, "__len__"):
                return len(sample)
            shape = getattr(sample, "shape", None)
            if callable(shape):
                shape = shape()
            try:
                dims = list(shape)
            except TypeError:
                dims = None
            if dims:
                return int(dims[0])
        raise GraphCaptureError(
            "len(Proxy) is not supported during graph capture; provide "
            "sample inputs to specialize on tensor shapes"
        )

    def __index__(self) -> int:
        raise GraphCaptureError("using a Proxy as an integer is not supported during graph capture")

    def __int__(self) -> int:
        raise GraphCaptureError("int(Proxy) is not supported during graph capture")

    def __float__(self) -> float:
        raise GraphCaptureError("float(Proxy) is not supported during graph capture")

    def __iter__(self):
        raise GraphCaptureError("iterating over a Proxy is not supported during graph capture")

    def __repr__(self) -> str:
        return f"Proxy({self.node.name})"


def _is_module(value: Any) -> bool:
    return callable(getattr(value, "forward", None)) and callable(
        getattr(value, "named_children", None)
    )


class Tracer:
    """Capture a callable into the canonical graph.

    This is intentionally a frontend primitive.  It is not part of the Stax
    backend and may later be replaced by a frame-evaluation frontend without
    changing the backend contract.

    Args:
        concrete_args: Optional mapping of argument name to a concrete value.
            Listed arguments are specialized away during capture: they do not
            become placeholders of the resulting graph.
    """

    def __init__(self, concrete_args: Optional[Dict[str, Any]] = None) -> None:
        self.graph = Graph()
        self.root: Any = None
        self.signature: Optional[inspect.Signature] = None
        self.concrete_args: Dict[str, Any] = dict(concrete_args or {})
        # Example values bound to placeholders during capture.  Metadata reads
        # (shape/dtype/device/...) resolve against them so Python control flow
        # can specialize statically; tensor DATA stays symbolic.
        self.sample_inputs: Dict[str, Any] = {}
        self._samples: Dict[str, Any] = {}
        # Placeholder-name -> attribute reads performed during capture
        # ({"shape", "len", "dtype", ...}).  Feeds dynamic-mode shape guards.
        self.metadata_touches: set[Tuple[str, str]] = set()
        # Qualified module path recorded per ``call_module`` node.  Modules
        # executed twice produce distinct ``path_0``/``path_1`` style entries,
        # mirroring torchvision's NodePathTracer.
        self.node_to_qualname: Dict[Node, str] = {}
        self._recorded_qualnames: set[str] = set()

    def is_leaf_module(self, module: Any, qualified_name: str) -> bool:
        """Return whether ``module`` should be traced as a single unit.

        The compiler frontend defaults to Dynamo-style behavior: every child
        module is inlined so backends receive primitive operations.  Frontends
        that need submodule boundaries in the graph (feature extraction)
        override this hook; returning ``True`` makes the tracer emit a
        ``call_module`` node targeting the module's qualified name instead of
        descending into ``forward``.
        """

        del module, qualified_name
        return False

    def _record_call_module(self, node: Node, qualified_name: str) -> None:
        if qualified_name not in self._recorded_qualnames:
            self._recorded_qualnames.add(qualified_name)
            self.node_to_qualname[node] = qualified_name
            return
        suffix = 0
        candidate = f"{qualified_name}_{suffix}"
        while candidate in self._recorded_qualnames:
            suffix += 1
            candidate = f"{qualified_name}_{suffix}"
        self._recorded_qualnames.add(candidate)
        self.node_to_qualname[node] = candidate

    def create_proxy(
        self,
        kind: str,
        target: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Proxy:
        return Proxy(self.graph.create_node(kind, target, args, kwargs), self)

    def trace(
        self, root: Any, sample_inputs: Optional[Dict[str, Any]] = None
    ) -> "GraphModule":
        self.root = root
        self.sample_inputs = dict(sample_inputs or {})
        if _is_module(root):
            function = root.forward
        elif callable(root):
            function = root
        else:
            raise TypeError(f"compile() expected a callable, got {type(root)!r}")

        self.signature = inspect.signature(function)
        parameters = list(self.signature.parameters.values())
        if any(
            parameter.kind
            in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            for parameter in parameters
        ):
            raise GraphCaptureError(
                "varargs and varkw arguments are not supported by this compiler frontend"
            )

        unknown_concrete = set(self.concrete_args) - {
            parameter.name for parameter in parameters
        }
        if unknown_concrete:
            raise GraphCaptureError(
                f"concrete arguments {sorted(unknown_concrete)} are not "
                "parameters of the traced callable"
            )

        values: Dict[str, Any] = {}
        for parameter in parameters:
            if parameter.name in self.concrete_args:
                # Specialized arguments never reach the graph.
                values[parameter.name] = self.concrete_args[parameter.name]
            else:
                placeholder_node = self.graph.placeholder(
                    parameter.name, parameter.default
                )
                sample = self.sample_inputs.get(parameter.name)
                if sample is not None:
                    self._samples[parameter.name] = sample
                values[parameter.name] = Proxy(placeholder_node, self)

        if _is_module(root):
            output = self._trace_module(root, function, parameters, values)
        else:
            output = self._invoke(function, parameters, values)

        self.graph.output(output)
        self.graph.lint()
        # Specialized (concrete) parameters disappear from the graph contract
        # together with their placeholders; every other parameter keeps its
        # kind and default so ``bind_partial().apply_defaults()`` keeps working.
        symbolic_parameters = [
            parameter for parameter in parameters if parameter.name not in self.concrete_args
        ]
        specialized_signature = inspect.Signature(symbolic_parameters)
        graph_module = GraphModule(root, self.graph, specialized_signature)
        if self._samples:
            graph_module.meta["sample_inputs"] = dict(self._samples)
        if self.metadata_touches:
            graph_module.meta["metadata_touches"] = sorted(self.metadata_touches)
        return graph_module

    @staticmethod
    def _invoke(
        function: Callable[..., Any],
        parameters: list[inspect.Parameter],
        values: Dict[str, Any],
    ) -> Any:
        """Call a traced function while preserving Python parameter kinds.

        Passing every symbolic parameter positionally breaks keyword-only
        arguments and changes the call contract before the backend ever sees
        the graph.  Dynamo/FX preserve the signature at this boundary; the
        small explicit dispatcher gives the same behavior for the canonical
        TensorPlay graph.  ``values`` may contain plain Python objects for
        arguments specialized through ``concrete_args`` alongside proxies.
        """

        positional: list[Any] = []
        keyword: dict[str, Any] = {}
        for parameter in parameters:
            value = values[parameter.name]
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                positional.append(value)
            elif parameter.kind is inspect.Parameter.KEYWORD_ONLY:
                keyword[parameter.name] = value
            else:
                raise GraphCaptureError(
                    "varargs and varkw arguments are not supported by this compiler frontend"
                )
        return function(*positional, **keyword)

    def _trace_module(
        self,
        root: Any,
        function: Callable[..., Any],
        parameters: list[inspect.Parameter],
        values: Dict[str, Any],
    ) -> Any:
        missing = object()
        patches: list[tuple[Any, str, Any]] = []

        def patch_attribute(module: Any, name: str, value: Any) -> None:
            previous = module.__dict__.get(name, missing)
            module.__dict__[name] = value
            patches.append((module, name, previous))

        def qualified(module_name: str, attribute: str) -> str:
            return f"{module_name}.{attribute}" if module_name else attribute

        try:
            # Child modules are inlined so the backend receives the operations
            # inside them, matching Dynamo's FX graph.  ``is_leaf_module`` may
            # opt a child out of inlining; such children become a single
            # ``call_module`` node targeting their qualified path, which is
            # what feature extraction needs to locate submodule outputs.
            leaf_qualnames: set[str] = set()
            for module_name, module in root.named_modules(remove_duplicate=True):
                for child_name, child in module.named_children():
                    qualname = qualified(module_name, child_name)

                    if self.is_leaf_module(child, qualname):
                        leaf_qualnames.add(qualname)

                        def call_module_child(
                            *args: Any,
                            _qualname: str = qualname,
                            **kwargs: Any,
                        ) -> Any:
                            proxy = self.create_proxy(
                                "call_module", _qualname, args, kwargs
                            )
                            self._record_call_module(proxy.node, _qualname)
                            return proxy

                        patch_attribute(module, child_name, call_module_child)
                    else:
                        def inline_child(
                            *args: Any, _child: Any = child, **kwargs: Any
                        ) -> Any:
                            return _child.forward(*args, **kwargs)

                        patch_attribute(module, child_name, inline_child)

            def under_leaf(module_qualname: str) -> bool:
                parts = module_qualname.split(".")
                return any(
                    ".".join(parts[: index + 1]) in leaf_qualnames
                    for index in range(len(parts))
                )

            # Parameters and buffers are graph attributes, not frozen Python
            # constants.  This keeps the compiled graph tied to the live
            # module state and preserves parameter autograd edges.  Parameters
            # below a leaf module stay untouched: the leaf is invoked as a
            # whole and its state must not appear as dangling graph inputs.
            for module_name, module in root.named_modules(remove_duplicate=True):
                if under_leaf(module_name):
                    continue
                for attribute_name, value in (
                    *getattr(module, "_parameters", {}).items(),
                    *getattr(module, "_buffers", {}).items(),
                ):
                    if value is None:
                        continue
                    if not hasattr(value, "shape") or not hasattr(value, "requires_grad"):
                        continue
                    patch_attribute(
                        module,
                        attribute_name,
                        self.create_proxy(
                            "get_attr",
                            qualified(module_name, attribute_name),
                            (),
                            {},
                        ),
                    )

            return self._invoke(function, parameters, values)
        finally:
            for module, name, previous in reversed(patches):
                if previous is missing:
                    module.__dict__.pop(name, None)
                else:
                    module.__dict__[name] = previous


class GraphModule:
    """Executable graph wrapper passed to compiler backends."""

    def __init__(
        self, root: Any, graph: Graph, signature: inspect.Signature
    ) -> None:
        self.root = root
        self.graph = graph
        self.signature = signature
        self.code = graph.python_code()
        self.meta: dict[str, Any] = {}
        self._compiled_forward: Optional[Callable[..., Any]] = None
        self._compiled_targets: dict[str, Any] = {}
        self._compiled_constants: list[Any] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self._compiled_forward is not None:
            return self._compiled_forward(*args, **kwargs)
        return self._interpret(*args, **kwargs)

    def recompile(self) -> Callable[..., Any]:
        """Generate an explicit Python executor for custom backend use.

        This is useful for frontend tests and deliberately opt-in fallback
        backends.  A performance backend must not use this executor: the
        ResNet benchmark requests ``strict_native`` and rejects it outright.
        """

        self._compiled_targets = {}
        self._compiled_constants = []
        lines = ["def _compiled(self, *args, **kwargs):"]
        lines.append("    _bound = self.signature.bind_partial(*args, **kwargs)")
        lines.append("    _bound.apply_defaults()")

        for node in self.graph.placeholders:
            lines.append(
                f"    {node.name} = _bound.arguments[{node.name!r}]"
            )

        for node in self.graph.nodes:
            if node.op in {"placeholder", "output"}:
                continue
            if node.op == "call_function":
                target_name = f"_target_{len(self._compiled_targets)}"
                self._compiled_targets[target_name] = self._resolve_target(node.target)
                args_expr = ", ".join(self._expr(arg) for arg in node.args)
                kwargs_expr = self._kwargs_expr(node.kwargs)
                call = f"{target_name}({args_expr}"
                if kwargs_expr:
                    call += f", {kwargs_expr}"
                call += ")"
                lines.append(f"    {node.name} = {call}")
            elif node.op == "call_method":
                resolved = list(node.args)
                if not resolved:
                    raise GraphCaptureError("call_method node has no receiver")
                receiver = self._expr(resolved[0])
                method_args = ", ".join(self._expr(arg) for arg in resolved[1:])
                kwargs_expr = self._kwargs_expr(node.kwargs)
                call = f"{receiver}.{node.target}({method_args}"
                if kwargs_expr:
                    if method_args:
                        call += ", "
                    call += kwargs_expr
                call += ")"
                lines.append(f"    {node.name} = {call}")
            elif node.op == "call_module":
                args_expr = ", ".join(self._expr(arg) for arg in node.args)
                kwargs_expr = self._kwargs_expr(node.kwargs)
                call = f"self._get_attr({node.target!r})({args_expr}"
                if kwargs_expr:
                    if args_expr:
                        call += ", "
                    call += kwargs_expr
                call += ")"
                lines.append(f"    {node.name} = {call}")
            elif node.op == "get_attr":
                lines.append(f"    {node.name} = self._get_attr({node.target!r})")
            else:
                raise GraphCaptureError(f"unsupported graph node kind: {node.op}")

        output_nodes = self.graph.outputs
        if not output_nodes:
            raise GraphCaptureError("graph has no output node")
        lines.append(f"    return {self._expr(output_nodes[-1].args[0])}")
        source = "\n".join(lines) + "\n"

        namespace: dict[str, Any] = {}
        exec(compile(source, "<tensorplay-compiled-graph>", "exec"), namespace)
        for name, target in self._compiled_targets.items():
            namespace[name] = target
        function = namespace["_compiled"]
        self._compiled_forward = types.MethodType(function, self)
        self.code = source
        return self.forward

    def _expr(self, value: Any) -> str:
        if isinstance(value, Node):
            return value.name
        if isinstance(value, tuple):
            items = ", ".join(self._expr(item) for item in value)
            if len(value) == 1:
                items += ","
            return f"({items})"
        if isinstance(value, list):
            return "[" + ", ".join(self._expr(item) for item in value) + "]"
        if isinstance(value, dict):
            items = ", ".join(
                f"{key!r}: {self._expr(item)}" for key, item in value.items()
            )
            return "{" + items + "}"
        if isinstance(value, slice):
            return (
                f"slice({self._expr(value.start)}, {self._expr(value.stop)}, "
                f"{self._expr(value.step)})"
            )
        if value is None or isinstance(value, (bool, int, float, str, bytes)):
            return repr(value)
        index = len(self._compiled_constants)
        self._compiled_constants.append(value)
        return f"self._compiled_constants[{index}]"

    def _kwargs_expr(self, kwargs: dict[str, Any]) -> str:
        if not kwargs:
            return ""
        return "**{" + ", ".join(
            f"{key!r}: {self._expr(value)}" for key, value in kwargs.items()
        ) + "}"

    def _interpret(self, *args: Any, _record_meta: bool = False, **kwargs: Any) -> Any:
        try:
            bound = self.signature.bind_partial(*args, **kwargs)
            bound.apply_defaults()
        except TypeError:
            raise

        def keep(node: Node, value: Any) -> Any:
            env[node] = value
            if _record_meta:
                node.meta["val"] = value
                shape = getattr(value, "shape", None)
                if shape is not None:
                    try:
                        node.meta["tensor_shape"] = tuple(int(d) for d in shape())
                    except (TypeError, ValueError):
                        try:
                            node.meta["tensor_shape"] = tuple(int(d) for d in shape)
                        except (TypeError, ValueError):
                            pass
            return value

        env: dict[Node, Any] = {}
        for node in self.graph.placeholders:
            if node.name not in bound.arguments:
                raise TypeError(f"missing required compiler input: {node.name}")
            keep(node, bound.arguments[node.name])

        for node in self.graph.nodes:
            if node.op == "placeholder":
                continue
            if node.op == "call_function":
                target = self._resolve_target(node.target)
                keep(node, target(
                    *self._resolve(node.args, env),
                    **self._resolve(node.kwargs, env),
                ))
            elif node.op == "call_method":
                resolved_args = self._resolve(node.args, env)
                receiver, *method_args = resolved_args
                keep(node, getattr(receiver, node.target)(*method_args, **self._resolve(node.kwargs, env)))
            elif node.op == "call_module":
                module = self._get_attr(node.target)
                keep(node, module(
                    *self._resolve(node.args, env),
                    **self._resolve(node.kwargs, env),
                ))
            elif node.op == "get_attr":
                keep(node, self._get_attr(node.target))
            elif node.op == "output":
                return self._resolve(node.args[0], env)
            else:
                raise GraphCaptureError(f"unsupported graph node kind: {node.op}")

        raise GraphCaptureError("graph has no output node")

    @staticmethod
    def _resolve(value: Any, env: dict[Node, Any]) -> Any:
        if isinstance(value, Node):
            return env[value]
        if isinstance(value, tuple):
            return tuple(GraphModule._resolve(item, env) for item in value)
        if isinstance(value, list):
            return [GraphModule._resolve(item, env) for item in value]
        if isinstance(value, dict):
            return {key: GraphModule._resolve(item, env) for key, item in value.items()}
        if isinstance(value, slice):
            return slice(
                GraphModule._resolve(value.start, env),
                GraphModule._resolve(value.stop, env),
                GraphModule._resolve(value.step, env),
            )
        return value

    def _get_attr(self, target: str) -> Any:
        value = self.root
        for part in target.split("."):
            value = getattr(value, part)
        return value

    @staticmethod
    def _resolve_target(target: Any) -> Any:
        if isinstance(target, Node):
            raise GraphCaptureError("calling a dynamically produced function is unsupported")
        return target


def dead_code_elimination(graph: Graph) -> int:
    """Remove pure nodes that cannot reach the graph output.

    Returns the number of removed nodes.  Placeholders are always kept so the
    calling contract of the graph stays intact.
    """

    live: set[Node] = set()
    worklist = list(graph.outputs)
    while worklist:
        node = worklist.pop()
        if node in live:
            continue
        live.add(node)
        worklist.extend(_iter_nodes(node.args))
        worklist.extend(_iter_nodes(node.kwargs))

    old_nodes = graph.nodes
    graph.nodes = [
        node
        for node in old_nodes
        if node in live or node.op == "placeholder"
    ]
    removed_count = len(old_nodes) - len(graph.nodes)
    if removed_count == 0:
        return 0

    for node in old_nodes:
        if node not in live and node.op != "placeholder":
            node.graph = None
    for node in graph.nodes:
        node.users.clear()
    for node in graph.nodes:
        for input_node in (*_iter_nodes(node.args), *_iter_nodes(node.kwargs)):
            input_node.users.add(node)
    return removed_count
