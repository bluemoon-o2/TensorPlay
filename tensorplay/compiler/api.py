"""Public compiler orchestration for TensorPlay.

The API follows the shape of ``torch.compile`` while keeping frontend
capture, backend selection, and execution concerns separate.  A backend is
never asked to discover Python control flow; it only receives a captured
``GraphModule`` and example inputs.
"""

from __future__ import annotations

import functools
import inspect
import threading
import weakref
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterable
from weakref import WeakSet

from .graph import GraphCaptureError, GraphModule, Tracer
from .passes import ConstFold, DeadCodeElimination, PassManager, ShapeProp
from .registry import CompilerFn, get_default_backend, lookup_backend


_compiling: ContextVar[bool] = ContextVar("tensorplay_compiling", default=False)
_capture_disabled: ContextVar[bool] = ContextVar(
    "tensorplay_capture_disabled", default=False
)
_compiled_wrappers: WeakSet[Any] = WeakSet()
_DEFAULT_RECOMPILE_LIMIT = 8


def is_compiling() -> bool:
    """Return whether Python is currently being captured by the compiler."""

    return _compiling.get() and not _capture_disabled.get()


@contextmanager
def _disable_capture() -> Iterable[None]:
    """Temporarily execute a region outside the active Stax capture.

    This is the TensorPlay counterpart of the capture boundary used by
    ``torch._disable_dynamo``.  The public compiler owns the capture state;
    optimizers only use this narrow internal context when Torch marks a
    stateful Python region as uncapturable.
    """

    token = _capture_disabled.set(True)
    try:
        yield
    finally:
        _capture_disabled.reset(token)


@contextmanager
def _compiler_context() -> Iterable[None]:
    token = _compiling.set(True)
    try:
        yield
    finally:
        _compiling.reset(token)


def _tensor_signature(value: Any, *, dynamic: bool) -> tuple[Any, ...] | None:
    module_name = type(value).__module__
    if not module_name.startswith("tensorplay"):
        return None
    shape = getattr(value, "shape", None)
    if callable(shape):
        shape = shape()
    try:
        shape = tuple(int(item) for item in shape)
        # Dynamic mode keeps rank specialization but removes concrete sizes.
        # This is the same useful boundary as TorchDynamo's first dynamic
        # shape policy; operations still receive the real runtime tensors.
        shape_key = ("dynamic", len(shape)) if dynamic else shape
    except (TypeError, ValueError):
        shape_key = repr(shape)
    dtype = getattr(value, "dtype", None)
    if callable(dtype):
        dtype = dtype()
    device = getattr(value, "device", None)
    if callable(device):
        device = device()
    requires_grad = getattr(value, "requires_grad", None)
    if callable(requires_grad):
        requires_grad = requires_grad()
    return (
        "tensor",
        type(value),
        shape_key,
        repr(dtype),
        repr(device),
        bool(requires_grad),
    )


def _value_signature(value: Any, *, dynamic: bool) -> Any:
    tensor_key = _tensor_signature(value, dynamic=dynamic)
    if tensor_key is not None:
        return tensor_key
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return (type(value), value)
    if isinstance(value, tuple):
        return (tuple, tuple(_value_signature(item, dynamic=dynamic) for item in value))
    if isinstance(value, list):
        return (list, tuple(_value_signature(item, dynamic=dynamic) for item in value))
    if isinstance(value, dict):
        items = sorted(
            (
                (
                    _value_signature(key, dynamic=dynamic),
                    _value_signature(item, dynamic=dynamic),
                )
                for key, item in value.items()
            ),
            key=repr,
        )
        return (dict, tuple(items))
    return (type(value), id(value))


def _input_signature(
    args: tuple[Any, ...], kwargs: dict[str, Any], *, dynamic: bool
) -> Any:
    return (
        tuple(_value_signature(item, dynamic=dynamic) for item in args),
        tuple(
            sorted(
                (key, _value_signature(value, dynamic=dynamic))
                for key, value in kwargs.items()
            )
        ),
    )


def _quick_value_signature(value: Any, *, dynamic: bool) -> Any:
    """Build the hot-path guard key without repr-heavy metadata formatting."""

    if type(value).__module__.startswith("tensorplay"):
        shape = getattr(value, "shape", None)
        try:
            shape_key = ("dynamic", len(shape)) if dynamic else tuple(int(item) for item in shape)
        except (TypeError, ValueError):
            shape_key = repr(shape)
        dtype = getattr(value, "dtype", None)
        device = getattr(value, "device", None)
        device_type = getattr(device, "type", None)
        if device_type is None:
            device_type = repr(device)
        device_key = (
            device_type,
            getattr(device, "index", None),
        )
        requires_grad = getattr(value, "requires_grad", False)
        return (type(value), shape_key, dtype, device_key, bool(requires_grad))
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return (type(value), value)
    if isinstance(value, tuple):
        return (tuple, tuple(_quick_value_signature(item, dynamic=dynamic) for item in value))
    if isinstance(value, list):
        return (list, tuple(_quick_value_signature(item, dynamic=dynamic) for item in value))
    if isinstance(value, dict):
        return (
            dict,
            tuple(
                sorted(
                    (
                        key,
                        _quick_value_signature(item, dynamic=dynamic),
                    )
                    for key, item in value.items()
                )
            ),
        )
    return (type(value), id(value))


def _quick_input_signature(
    args: tuple[Any, ...], kwargs: dict[str, Any], *, dynamic: bool
) -> Any:
    return (
        tuple(_quick_value_signature(item, dynamic=dynamic) for item in args),
        tuple(
            sorted(
                (key, _quick_value_signature(value, dynamic=dynamic))
                for key, value in kwargs.items()
            )
        ),
    )


def _backend_kwargs(
    *,
    mode: str | None,
    options: dict[str, Any] | None,
    name: str | None,
    dynamic: bool | None,
    strict_native: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if mode is not None and mode != "default":
        kwargs["mode"] = mode
    if options:
        kwargs["options"] = dict(options)
    if name is not None:
        kwargs["name"] = name
    if dynamic is not None:
        kwargs["dynamic"] = dynamic
    if strict_native:
        # Capture may run in Python, but the returned callable must be a
        # native backend executable when this contract is requested.
        kwargs["strict_native"] = True
    return kwargs


def compile(
    model: Callable[..., Any] | None = None,
    *,
    fullgraph: bool = False,
    dynamic: bool | None = None,
    backend: str | CompilerFn | None = None,
    mode: str | None = None,
    options: dict[str, Any] | None = None,
    name: str | None = None,
    disable: bool = False,
    recompile_limit: int | None = None,
    isolate_recompiles: bool = False,
    strict_native: bool = False,
    dynamic_shapes: Any = None,
) -> Callable[..., Any]:
    """Compile a callable through the TensorPlay compiler frontend.

    ``backend`` may be a registered name or a callable with the contract
    ``backend(graph_module, example_inputs, **kwargs) -> callable``.  The
    frontend caches specializations by input metadata and falls back to the
    original callable when a non-``fullgraph`` capture cannot represent the
    Python region.
    """

    normalized_dynamic_shapes = dynamic_shapes
    if dynamic_shapes is not None:
        if dynamic is not None:
            raise RuntimeError("dynamic and dynamic_shapes cannot both be specified")
        if not isinstance(dynamic_shapes, bool):
            raise TypeError(
                "TensorPlay dynamic_shapes currently accepts only a bool; "
                "use a bool dynamic policy for this frontend"
            )
        dynamic = dynamic_shapes

    if mode is not None and options is not None:
        raise RuntimeError("Either mode or options can be specified, but not both")
    if mode is None and options is None:
        mode = "default"
    if options is not None and not isinstance(options, dict):
        raise TypeError(f"options must be a dict, got {type(options)!r}")

    if model is None:
        return lambda actual_model: compile(
            actual_model,
            fullgraph=fullgraph,
            backend=backend,
            dynamic=dynamic,
            mode=mode,
            options=options,
            name=name,
            disable=disable,
            recompile_limit=recompile_limit,
            isolate_recompiles=isolate_recompiles,
            strict_native=strict_native,
            # ``dynamic_shapes`` has already been normalized into ``dynamic``
            # above.  Passing it again would look like the user supplied both
            # mutually exclusive knobs on the recursive decorator call.
            dynamic_shapes=None,
        )
    if not callable(model):
        raise TypeError(f"compile() expected a callable, got {type(model)!r}")
    if disable:
        return model
    if recompile_limit is not None and recompile_limit < 1:
        raise ValueError("recompile_limit must be positive")

    backend_spec = get_default_backend() if backend is None else backend
    compiler_fn = lookup_backend(backend_spec)
    backend_kwargs = _backend_kwargs(
        mode=mode,
        options=options,
        name=name,
        dynamic=dynamic,
        strict_native=strict_native,
    )
    specialization_dynamic = dynamic is True
    specialization_limit = (
        _DEFAULT_RECOMPILE_LIMIT if recompile_limit is None else recompile_limit
    )
    cache: dict[Any, Callable[..., Any]] = {}
    lock = threading.RLock()
    last_quick_key: Any = object()
    last_compiled_fn: Callable[..., Any] | None = None
    last_arg_refs: tuple[weakref.ReferenceType[Any], ...] | None = None
    guard_param_names: tuple[str, ...] = ()

    def _guard_component(
        args_: tuple[Any, ...],
        kwargs_: dict[str, Any],
        builder: Callable[..., Any],
    ) -> tuple[Any, ...]:
        if not guard_param_names:
            return ()
        target = model.forward if _is_module_like(model) else model
        try:
            bound = inspect.signature(target).bind_partial(*args_, **kwargs_)
            bound.apply_defaults()
        except (TypeError, ValueError):
            return ("shape-guards", "unbound")
        return (
            "shape-guards",
            tuple(
                builder(bound.arguments.get(name), dynamic=False)
                for name in guard_param_names
            ),
        )

    @functools.wraps(model)
    def optimized(*args: Any, **kwargs: Any) -> Any:
        nonlocal last_quick_key, last_compiled_fn, last_arg_refs, guard_param_names
        same_last_args = (
            not kwargs
            and last_arg_refs is not None
            and len(last_arg_refs) == len(args)
            and all(reference() is value for reference, value in zip(last_arg_refs, args))
        )
        quick_key = (
            last_quick_key
            if same_last_args
            else (
                _quick_input_signature(
                    args, kwargs, dynamic=specialization_dynamic
                ),
                _guard_component(args, kwargs, _quick_value_signature),
            )
        )
        with lock:
            if cache and last_compiled_fn is not None and quick_key == last_quick_key:
                compiled_fn = last_compiled_fn
            else:
                key = (
                    _input_signature(args, kwargs, dynamic=specialization_dynamic),
                    _guard_component(args, kwargs, _value_signature),
                )
                compiled_fn = cache.get(key)
            if compiled_fn is None:
                if len(cache) >= specialization_limit:
                    if fullgraph:
                        raise RuntimeError(
                            "TensorPlay compile specialization limit reached in fullgraph mode"
                        )
                    compiled_fn = model
                    cache[key] = compiled_fn
                else:
                    compiled_fn, captured_gm = _compile_region(
                        model,
                        compiler_fn,
                        args,
                        kwargs,
                        fullgraph=fullgraph,
                        backend_kwargs=backend_kwargs,
                    )
                    promoted = _extract_shape_guard_params(captured_gm)
                    if promoted - set(guard_param_names):
                        # Keys gain a guard component once shape reads exist;
                        # invalidate so every entry is stored uniformly.
                        guard_param_names = tuple(
                            sorted({*guard_param_names, *promoted})
                        )
                        cache.clear()
                        last_compiled_fn = None
                    key = (
                        _input_signature(
                            args, kwargs, dynamic=specialization_dynamic
                        ),
                        _guard_component(args, kwargs, _value_signature),
                    )
                    cache[key] = compiled_fn
            last_quick_key = quick_key
            last_compiled_fn = compiled_fn
            if not kwargs:
                try:
                    last_arg_refs = tuple(weakref.ref(value) for value in args)
                except TypeError:
                    last_arg_refs = None
            else:
                last_arg_refs = None
        return compiled_fn(*args, **kwargs)

    optimized._tensorplay_backend = backend_spec  # type: ignore[attr-defined]
    optimized._tensorplay_cache = cache  # type: ignore[attr-defined]
    optimized._tensorplay_original = model  # type: ignore[attr-defined]
    optimized._tensorplay_dynamic = dynamic  # type: ignore[attr-defined]
    optimized._tensorplay_dynamic_shapes = normalized_dynamic_shapes  # type: ignore[attr-defined]
    optimized._tensorplay_isolate_recompiles = isolate_recompiles  # type: ignore[attr-defined]
    optimized._tensorplay_recompile_limit = specialization_limit  # type: ignore[attr-defined]
    _compiled_wrappers.add(optimized)
    return optimized


def _bind_sample_arguments(
    model: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any] | None:
    """Bind example call arguments to parameter names for the tracer.

    Metadata reads on placeholders (``x.shape[0] > 2``, ``range(x.ndim)``)
    then specialize statically during capture; the compile signature already
    keys on these fields, so no additional recompile conditions appear.
    """

    target = model.forward if _is_module_like(model) else model
    try:
        signature = inspect.signature(target)
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
    except (TypeError, ValueError):
        return None
    return dict(bound.arguments)


def _is_module_like(value: Any) -> bool:
    return hasattr(value, "named_modules") and callable(
        getattr(value, "forward", None)
    )


def _compile_region(
    model: Callable[..., Any],
    compiler_fn: CompilerFn,
    example_inputs: tuple[Any, ...],
    example_kwargs: dict[str, Any],
    *,
    fullgraph: bool,
    backend_kwargs: dict[str, Any],
) -> Callable[..., Any]:
    try:
        with _compiler_context():
            graph_module = Tracer().trace(
                model,
                sample_inputs=_bind_sample_arguments(
                    model, example_inputs, example_kwargs
                ),
            )
            # Default capture pipeline: constant folding first, then dead
            # code elimination.  Backends always receive a folded, linted
            # graph; ShapeProp below additionally annotates tensor shapes.
            PassManager([ConstFold(), DeadCodeElimination()])(graph_module)
    except GraphCaptureError:
        if fullgraph:
            raise
        return model

    # Backend failures are compiler failures, not graph breaks.  In
    # particular, a Stax lowering error must not silently turn a requested
    # compiled region into an uncompiled call.
    # FX/Inductor backends receive graph inputs in placeholder order, including
    # values supplied through keywords and defaults.  Passing only positional
    # arguments makes a keyword-only/scalar placeholder appear to be missing
    # and is especially harmful for native Stax lowering.
    bound = graph_module.signature.bind_partial(*example_inputs, **example_kwargs)
    bound.apply_defaults()
    backend_inputs = [bound.arguments[node.name] for node in graph_module.graph.placeholders]

    # Advisory shape/value metadata for backends and visualization; never a
    # reason to reject an otherwise compilable region.
    try:
        ShapeProp(backend_inputs)(graph_module)
    except (GraphCaptureError, RuntimeError):
        pass

    with _compiler_context():
        compiled = compiler_fn(graph_module, backend_inputs, **backend_kwargs)

    if not callable(compiled):
        raise TypeError(
            f"compiler backend returned {type(compiled)!r}; expected a callable"
        )
    return compiled, graph_module


_SHAPE_GUARD_ATTRS = frozenset({"shape", "len", "ndim"})


def _extract_shape_guard_params(graph_module: GraphModule) -> frozenset[str]:
    """Parameters whose captured metadata reads require exact-shape guards.

    Branching on ``x.shape[0]`` bakes one side of the branch into the graph,
    so a dynamic-mode cache entry may only be reused while that placeholder's
    shape stays identical.  dtype/device/reads need no extra guards: they are
    already part of every specialization signature.
    """

    touches = getattr(graph_module, "meta", {}).get("metadata_touches") or ()
    names = {name for name, attr in touches if attr in _SHAPE_GUARD_ATTRS}
    if not names or graph_module.signature is None:
        return frozenset()
    return frozenset(
        name for name in graph_module.signature.parameters if name in names
    )


def reset() -> None:
    """Clear all per-wrapper compiler specializations."""

    for wrapper in list(_compiled_wrappers):
        cache = getattr(wrapper, "_tensorplay_cache", None)
        if cache is not None:
            cache.clear()
