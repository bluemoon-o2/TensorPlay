"""CUDA graphs orchestration (L5-M3).

Management layer modeled on ``torch/_inductor/cudagraph_trees.py`` concepts
(capture once, replay with static buffers), kept decoupled from the native
runtime through an injectable binding surface so the full logic is testable
before the system layer lands.

Required native surface (probed lazily on ``tensorplay._C``):

===============================  ==========================================
symbol                           semantics
===============================  ==========================================
``cuda_graph_begin_capture()``   start capture on the dedicated side stream
                                 (it becomes the thread's current stream)
``cuda_graph_end_capture()``     stop -> opaque graph handle
``cuda_graph_instantiate(g)``    compile to executable
``cuda_graph_launch(e)``         enqueue executable on current stream
===============================  ==========================================

Optional symbols used when present (all shipped by ``tensorplay._C`` builds
with CUDA): ``cuda_graph_capture_stream()`` exposes the dedicated capture
side stream so warmup runs on the same stream as capture (lazy per-stream
state such as cuBLAS workspaces must see both equally);
``cuda_stream_get_current()/cuda_stream_set_current(s)`` save and restore the
caller's stream around the whole sequence.  Allocations issued during
capture are routed into a graph-private allocator pool natively, keeping
replay-baked addresses exclusive until the entry is dropped.

Until the required symbols land, :meth:`CudaGraphManager.capture` raises
:class:`NotImplementedError` naming the missing symbols. Tests may inject a
fake via ``CudaGraphManager(native=...)``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


class CudaGraphError(RuntimeError):
    """Raised for capture/replay contract violations."""


_REQUIRED_NATIVE = (
    "cuda_graph_begin_capture",
    "cuda_graph_end_capture",
    "cuda_graph_instantiate",
    "cuda_graph_launch",
)


def _default_native() -> Any:
    try:
        from .. import _C  # type: ignore
    except Exception as exc:  # pragma: no cover - import failure diagnostics
        raise NotImplementedError(
            "CUDA graph support requires tensorplay._C; import failed: "
            f"{exc!r}. Required symbols: {', '.join(_REQUIRED_NATIVE)}"
        ) from exc
    missing = [name for name in _REQUIRED_NATIVE if not hasattr(_C, name)]
    if missing:
        raise NotImplementedError(
            "CUDA graph bindings not implemented yet in tensorplay._C: "
            f"{', '.join(missing)}. See module docstring for the required "
            "native surface contract."
        )
    return _C


def _shape_signature(args: Sequence[Any]) -> Tuple:
    return tuple(
        (tuple(getattr(a, "shape", ())), str(getattr(a, "dtype", "")))
        for a in args
    )


def _clone_static(tensor: Any) -> Any:
    """Allocate the static input buffer as an exact copy of ``tensor``."""

    clone = tensor.clone()
    return clone


def _switch_to_capture_stream(native: Any) -> Any:
    """Move the calling thread onto the native capture side stream.

    Returns the caller's previous stream when the switch happened (the
    caller must restore it), or ``None`` when the native surface lacks the
    optional stream symbols and capture will run on whatever is current.
    """

    get_capture_stream = getattr(native, "cuda_graph_capture_stream", None)
    set_current = getattr(native, "cuda_stream_set_current", None)
    if get_capture_stream is None or set_current is None:
        return None
    side_stream = get_capture_stream()
    get_current = getattr(native, "cuda_stream_get_current", None)
    previous = get_current() if get_current is not None else None
    set_current(side_stream)
    return previous


def _copy_into(dst: Any, src: Any) -> None:
    copy = getattr(dst, "copy_", None)
    if copy is None:
        raise CudaGraphError(
            "static input buffer lacks copy_; cannot stage replay inputs"
        )
    copy(src)


class _GraphEntry:
    def __init__(self, key: str, signature: Tuple, handle: Any,
                 static_inputs: List[Any], static_outputs: List[Any]) -> None:
        self.key = key
        self.signature = signature
        self.handle = handle
        self.static_inputs = static_inputs
        self.static_outputs = static_outputs
        self.replays = 0


class CudaGraphManager:
    """Capture functions once, replay them against static buffers."""

    def __init__(self, native: Optional[Any] = None, max_entries: int = 8) -> None:
        self._native = native if native is not None else None
        self._owns_lookup = native is None
        self._entries: Dict[str, _GraphEntry] = {}
        self.max_entries = max_entries
        self.capturing: Optional[str] = None

    # -- native plumbing ------------------------------------------------------

    @property
    def native(self) -> Any:
        if self._native is None:
            self._native = _default_native()
        return self._native

    # -- API ------------------------------------------------------------------

    def capture(self, key: str, fn: Callable[..., Any], *sample_args: Any) -> _GraphEntry:
        if self.capturing is not None:
            raise CudaGraphError(
                f"nested capture attempted ({self.capturing!r} already active)"
            )
        existing = self._entries.get(key)
        signature = _shape_signature(sample_args)
        if existing is not None:
            if existing.signature != signature:
                raise CudaGraphError(
                    f"entry {key!r} was captured for {existing.signature}, "
                    f"refusing re-capture for {signature}; use a new key"
                )
            return existing
        if len(self._entries) >= self.max_entries:
            raise CudaGraphError(
                f"graph cache full ({self.max_entries}); clear stale entries"
            )

        native = self.native
        # Warmup must run on the same stream capture uses (lazy per-stream
        # state such as cuBLAS workspaces would otherwise land mid-capture),
        # so switch to the dedicated capture side stream when the native
        # surface exposes it.  begin/end capture manage the current stream
        # themselves for the captured window; we restore the caller's stream
        # afterwards so replays enqueue where the user expects.
        restore_stream = _switch_to_capture_stream(native)
        try:
            # Warmup executes lazy initialisations outside capture.
            fn(*sample_args)
            self.capturing = key
            native.cuda_graph_begin_capture()
            static_inputs = [_clone_static(a) for a in sample_args]
            outputs = fn(*static_inputs)
            graph = native.cuda_graph_end_capture()
            executable = native.cuda_graph_instantiate(graph)
        finally:
            self.capturing = None
            if restore_stream is not None:
                native.cuda_stream_set_current(restore_stream)
        out_list = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
        entry = _GraphEntry(key, signature, executable, static_inputs, out_list)
        self._entries[key] = entry
        return entry

    def replay(self, key: str, *args: Any) -> List[Any]:
        entry = self._entries.get(key)
        if entry is None:
            raise CudaGraphError(f"no captured graph under key {key!r}")
        if len(args) != len(entry.static_inputs):
            raise CudaGraphError(
                f"entry {key!r} expects {len(entry.static_inputs)} inputs, got {len(args)}"
            )
        signature = _shape_signature(args)
        if signature != entry.signature:
            raise CudaGraphError(
                f"entry {key!r} captured for {entry.signature}, replay args are {signature}"
            )
        for dst, src in zip(entry.static_inputs, args):
            _copy_into(dst, src)
        self.native.cuda_graph_launch(entry.handle)
        entry.replays += 1
        return list(entry.static_outputs)

    def clear(self, key: Optional[str] = None) -> None:
        if key is None:
            self._entries.clear()
        else:
            self._entries.pop(key, None)
