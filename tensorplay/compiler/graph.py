"""The canonical TensorPlay compiler graph.

This module is deliberately independent of Stax.  A compiler frontend owns
capture and produces this graph; backends consume :class:`GraphModule`.
Keeping the graph here mirrors the PyTorch split between Dynamo/FX and a
backend such as Inductor.
"""

from __future__ import annotations

import inspect
import operator
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


class Graph:
    """A mutable, topologically ordered operation graph."""

    def __init__(self) -> None:
        self.nodes: list[Node] = []
        self._counter = 0

    @property
    def placeholders(self) -> list[Node]:
        return [node for node in self.nodes if node.op == "placeholder"]

    @property
    def outputs(self) -> list[Node]:
        return [node for node in self.nodes if node.op == "output"]

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
        node_name = name or f"_{self._counter}"
        self._counter += 1
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

    def output(self, value: Any) -> Node:
        return self.create_node("output", "output", (value,))

    def lint(self) -> None:
        positions = {node: index for index, node in enumerate(self.nodes)}
        for node in self.nodes:
            for input_node in (*_iter_nodes(node.args), *_iter_nodes(node.kwargs)):
                if positions[input_node] >= positions[node]:
                    raise GraphCaptureError(
                        f"Graph is not topologically ordered: {input_node} -> {node}"
                    )

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

    def _property(self, name: str) -> "Proxy":
        return self.tracer.create_proxy("call_function", getattr, (self, name), {})

    @property
    def shape(self) -> "Proxy":
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
        raise GraphCaptureError("len(Proxy) is not supported during graph capture")

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
    """

    def __init__(self) -> None:
        self.graph = Graph()
        self.root: Any = None
        self.signature: Optional[inspect.Signature] = None

    def create_proxy(
        self,
        kind: str,
        target: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Proxy:
        return Proxy(self.graph.create_node(kind, target, args, kwargs), self)

    def trace(self, root: Any) -> "GraphModule":
        self.root = root
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

        proxies = [
            Proxy(self.graph.placeholder(parameter.name, parameter.default), self)
            for parameter in parameters
        ]

        if _is_module(root):
            output = self._trace_module(root, function, proxies)
        else:
            output = self._invoke(function, parameters, proxies)

        self.graph.output(output)
        self.graph.lint()
        return GraphModule(root, self.graph, self.signature)

    @staticmethod
    def _invoke(
        function: Callable[..., Any],
        parameters: list[inspect.Parameter],
        proxies: list[Proxy],
    ) -> Any:
        """Call a traced function while preserving Python parameter kinds.

        Passing every symbolic parameter positionally breaks keyword-only
        arguments and changes the call contract before the backend ever sees
        the graph.  Dynamo/FX preserve the signature at this boundary; the
        small explicit dispatcher gives the same behavior for the canonical
        TensorPlay graph.
        """

        positional: list[Proxy] = []
        keyword: dict[str, Proxy] = {}
        for parameter, proxy in zip(parameters, proxies):
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                positional.append(proxy)
            elif parameter.kind is inspect.Parameter.KEYWORD_ONLY:
                keyword[parameter.name] = proxy
            else:
                raise GraphCaptureError(
                    "varargs and varkw arguments are not supported by this compiler frontend"
                )
        return function(*positional, **keyword)

    def _trace_module(
        self, root: Any, function: Callable[..., Any], proxies: list[Proxy]
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
            # Inline child forwards so the backend receives the operations in
            # the module, matching Dynamo's FX graph rather than a Python
            # ``call_module`` escape hatch.
            for module_name, module in root.named_modules(remove_duplicate=True):
                for child_name, child in module.named_children():
                    def inline_child(
                        *args: Any, _child: Any = child, **kwargs: Any
                    ) -> Any:
                        return _child.forward(*args, **kwargs)

                    patch_attribute(module, child_name, inline_child)

            # Parameters and buffers are graph attributes, not frozen Python
            # constants.  This keeps the compiled graph tied to the live
            # module state and preserves parameter autograd edges.
            for module_name, module in root.named_modules(remove_duplicate=True):
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

            return self._invoke(
                function,
                list(self.signature.parameters.values()),
                proxies,
            )
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

    def _interpret(self, *args: Any, **kwargs: Any) -> Any:
        try:
            bound = self.signature.bind_partial(*args, **kwargs)
            bound.apply_defaults()
        except TypeError:
            raise

        env: dict[Node, Any] = {}
        for node in self.graph.placeholders:
            if node.name not in bound.arguments:
                raise TypeError(f"missing required compiler input: {node.name}")
            env[node] = bound.arguments[node.name]

        for node in self.graph.nodes:
            if node.op == "placeholder":
                continue
            if node.op == "call_function":
                target = self._resolve_target(node.target)
                env[node] = target(
                    *self._resolve(node.args, env),
                    **self._resolve(node.kwargs, env),
                )
            elif node.op == "call_method":
                resolved_args = self._resolve(node.args, env)
                receiver, *method_args = resolved_args
                env[node] = getattr(receiver, node.target)(*method_args, **self._resolve(node.kwargs, env))
            elif node.op == "call_module":
                module = self._get_attr(node.target)
                env[node] = module(
                    *self._resolve(node.args, env),
                    **self._resolve(node.kwargs, env),
                )
            elif node.op == "get_attr":
                env[node] = self._get_attr(node.target)
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


def dead_code_elimination(graph: Graph) -> bool:
    """Remove pure nodes that cannot reach the graph output."""

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
    if len(old_nodes) == len(graph.nodes):
        return False

    for node in graph.nodes:
        node.users.clear()
    for node in graph.nodes:
        for input_node in (*_iter_nodes(node.args), *_iter_nodes(node.kwargs)):
            input_node.users.add(node)
    return True
