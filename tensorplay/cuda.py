"""CUDA runtime controls with PyTorch-compatible stream/event semantics."""

from __future__ import annotations

from typing import Any, Optional, Union

from ._C import Device, DeviceType
from ._C import _cuda as _cuda_core


is_available = _cuda_core.is_available
device_count = _cuda_core.device_count
empty_cache = _cuda_core.empty_cache

_initialized = False


def _require_cuda() -> None:
    if not is_available():
        raise RuntimeError("CUDA is not available in this TensorPlay runtime")


def _get_device_index(value: Optional[Union[int, str, Device, Any]], optional: bool = True) -> int:
    if value is None:
        if optional:
            return current_device()
        raise ValueError("Expected a CUDA device, but got None")
    if isinstance(value, int):
        index = value
    elif isinstance(value, str):
        parsed = Device(value)
        if not parsed.is_cuda():
            raise ValueError(f"Expected a CUDA device, but got {value!r}")
        index = parsed.index
    elif isinstance(value, Device):
        if not value.is_cuda():
            raise ValueError(f"Expected a CUDA device, but got {value}")
        index = value.index
    elif hasattr(value, "device") and isinstance(value.device, Device):
        return _get_device_index(value.device, optional=optional)
    elif hasattr(value, "index"):
        index = int(value.index)
    else:
        raise TypeError(f"Invalid CUDA device: {value!r}")
    if index < 0:
        return current_device() if optional else index
    return int(index)


def init() -> None:
    """Initialize CUDA state. TensorPlay otherwise initializes it lazily."""
    global _initialized
    _require_cuda()
    _cuda_core.current_device()
    _initialized = True


def is_initialized() -> bool:
    return _initialized


def current_device() -> int:
    _require_cuda()
    return _cuda_core.current_device()


def set_device(value: Union[int, str, Device, Any]) -> None:
    _require_cuda()
    index = _get_device_index(value, optional=False)
    if index >= 0:
        _cuda_core.set_device(index)


class device:
    """Context manager that selects a CUDA device and restores it on exit."""

    def __init__(self, value: Union[int, str, Device, Any]):
        self.idx = _get_device_index(value, optional=False)
        self.prev_idx = -1

    def __enter__(self) -> "device":
        self.prev_idx = current_device()
        if self.idx >= 0 and self.idx != self.prev_idx:
            _cuda_core.set_device(self.idx)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.prev_idx >= 0 and self.prev_idx != self.idx:
            _cuda_core.set_device(self.prev_idx)


class device_of(device):
    """Select the device of a CUDA tensor or stream; CPU objects are a no-op."""

    def __init__(self, obj: Any):
        obj_device = getattr(obj, "device", None)
        if isinstance(obj_device, Device) and obj_device.is_cuda():
            super().__init__(obj_device)
        else:
            self.idx = -1
            self.prev_idx = -1

    def __enter__(self) -> "device_of":
        if self.idx >= 0:
            super().__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.idx >= 0:
            super().__exit__(exc_type, exc_value, traceback)


def get_device_name(value: Optional[Union[int, str, Device, Any]] = None) -> str:
    return _cuda_core.get_device_name(_get_device_index(value))


def get_device_capability(value: Optional[Union[int, str, Device, Any]] = None) -> tuple[int, int]:
    return _cuda_core.get_device_capability(_get_device_index(value))


def get_device_properties(value: Optional[Union[int, str, Device, Any]] = None):
    return _cuda_core.get_device_properties(_get_device_index(value))


def synchronize(value: Optional[Union[int, str, Device, Any]] = None) -> None:
    _require_cuda()
    _cuda_core.synchronize(_get_device_index(value))


def memory_allocated(value: Optional[Union[int, str, Device, Any]] = None) -> int:
    return _cuda_core.memory_allocated(_get_device_index(value))


def memory_reserved(value: Optional[Union[int, str, Device, Any]] = None) -> int:
    return _cuda_core.memory_reserved(_get_device_index(value))


def max_memory_allocated(value: Optional[Union[int, str, Device, Any]] = None) -> int:
    return _cuda_core.max_memory_allocated(_get_device_index(value))


def max_memory_reserved(value: Optional[Union[int, str, Device, Any]] = None) -> int:
    return _cuda_core.max_memory_reserved(_get_device_index(value))


def reset_peak_memory_stats(value: Optional[Union[int, str, Device, Any]] = None) -> None:
    _cuda_core.reset_peak_memory_stats(_get_device_index(value))


def reset_max_memory_allocated(value: Optional[Union[int, str, Device, Any]] = None) -> None:
    # Kept for compatibility with older TensorPlay and PyTorch code.
    reset_peak_memory_stats(value)


def reset_max_memory_reserved(value: Optional[Union[int, str, Device, Any]] = None) -> None:
    reset_peak_memory_stats(value)


def memory_stats(value: Optional[Union[int, str, Device, Any]] = None) -> dict[str, int]:
    index = _get_device_index(value)
    allocated = _cuda_core.memory_allocated(index)
    reserved = _cuda_core.memory_reserved(index)
    max_allocated = _cuda_core.max_memory_allocated(index)
    max_reserved = _cuda_core.max_memory_reserved(index)
    return {
        "allocated_bytes.all.current": allocated,
        "allocated_bytes.all.peak": max_allocated,
        "reserved_bytes.all.current": reserved,
        "reserved_bytes.all.peak": max_reserved,
    }


def memory_summary(value: Optional[Union[int, str, Device, Any]] = None, abbreviated: bool = False) -> str:
    index = _get_device_index(value)
    stats = memory_stats(index)
    mib = 1024 * 1024
    return (
        f"TensorPlay CUDA memory summary, device cuda:{index}\n"
        f"Allocated: {stats['allocated_bytes.all.current'] / mib:.2f} MiB "
        f"(peak {stats['allocated_bytes.all.peak'] / mib:.2f} MiB)\n"
        f"Reserved:  {stats['reserved_bytes.all.current'] / mib:.2f} MiB "
        f"(peak {stats['reserved_bytes.all.peak'] / mib:.2f} MiB)"
    )


class Event:
    """A lazily-created CUDA event."""

    def __init__(self, enable_timing: bool = False, blocking: bool = False,
                 interprocess: bool = False, *, _event=None):
        _require_cuda()
        self._event = _event or _cuda_core._CudaEvent(
            enable_timing, blocking, interprocess
        )

    @property
    def device(self):
        return self._event.device

    @property
    def cuda_event(self) -> int:
        return self._event.cuda_event

    def record(self, stream: Optional["Stream"] = None) -> None:
        core_stream = None if stream is None else _unwrap_stream(stream)
        self._event.record(core_stream)

    def wait(self, stream: Optional["Stream"] = None) -> None:
        core_stream = None if stream is None else _unwrap_stream(stream)
        self._event.wait(core_stream)

    def query(self) -> bool:
        return self._event.query()

    def elapsed_time(self, end_event: "Event") -> float:
        if not isinstance(end_event, Event):
            raise TypeError("end_event must be a tensorplay.cuda.Event")
        return self._event.elapsed_time(end_event._event)

    def synchronize(self) -> None:
        self._event.synchronize()

    def __repr__(self) -> str:
        return repr(self._event)


class Stream:
    """A CUDA stream acquired from TensorPlay's per-device stream pool."""

    def __init__(self, device: Optional[Union[int, str, Device, Any]] = None,
                 priority: int = 0, *, _stream=None, **kwargs):
        _require_cuda()
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected Stream argument(s): {unknown}")
        self._stream = _stream or _cuda_core.get_stream_from_pool(
            int(priority), _get_device_index(device)
        )
        self._contexts: list[_StreamContext] = []

    @classmethod
    def _from_core(cls, core_stream) -> "Stream":
        return cls(_stream=core_stream)

    @property
    def device(self) -> Device:
        return self._stream.device

    @property
    def cuda_stream(self) -> int:
        return self._stream.cuda_stream

    @property
    def priority(self) -> int:
        return self._stream.priority

    def query(self) -> bool:
        return self._stream.query()

    def wait_event(self, event: Event) -> None:
        if not isinstance(event, Event):
            raise TypeError("event must be a tensorplay.cuda.Event")
        self._stream.wait_event(event._event)

    def wait_stream(self, other: "Stream") -> None:
        self._stream.wait_stream(_unwrap_stream(other))

    def record_event(self, event: Optional[Event] = None) -> Event:
        if event is None:
            event = Event()
        event.record(self)
        return event

    def synchronize(self) -> None:
        self._stream.synchronize()

    def __enter__(self) -> "Stream":
        context = _StreamContext(self)
        self._contexts.append(context)
        context.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._contexts.pop().__exit__(exc_type, exc_value, traceback)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Stream) and self._stream == other._stream

    def __hash__(self) -> int:
        return hash((self.device.index, self.cuda_stream))

    def __repr__(self) -> str:
        return repr(self._stream)


def _unwrap_stream(value: Stream):
    if not isinstance(value, Stream):
        raise TypeError("stream must be a tensorplay.cuda.Stream")
    return value._stream


def current_stream(device: Optional[Union[int, str, Device, Any]] = None) -> Stream:
    _require_cuda()
    return Stream._from_core(_cuda_core.current_stream(_get_device_index(device)))


def default_stream(device: Optional[Union[int, str, Device, Any]] = None) -> Stream:
    _require_cuda()
    return Stream._from_core(_cuda_core.default_stream(_get_device_index(device)))


def set_stream(value: Optional[Stream]) -> None:
    _require_cuda()
    if value is None:
        value = default_stream()
    _cuda_core.set_stream(_unwrap_stream(value))


class _StreamContext:
    def __init__(self, value: Optional[Stream]):
        self.stream = value
        self.prev_device = -1
        self.prev_stream: Optional[Stream] = None

    def __enter__(self):
        if self.stream is None:
            return None
        self.prev_device = current_device()
        destination = self.stream.device.index
        self.prev_stream = current_stream(destination)
        if destination != self.prev_device:
            _cuda_core.set_device(destination)
        set_stream(self.stream)
        return self.stream

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.stream is None:
            return
        if self.prev_stream is not None:
            set_stream(self.prev_stream)
        if self.prev_device >= 0 and self.prev_device != self.stream.device.index:
            _cuda_core.set_device(self.prev_device)


def stream(value: Optional[Stream]) -> _StreamContext:
    """Context manager selecting ``value`` and restoring device/stream state."""
    return _StreamContext(value)


def get_stream_priority_range() -> tuple[int, int]:
    _require_cuda()
    return _cuda_core.get_stream_priority_range()


def _sleep(cycles: int) -> None:
    """Busy-wait on the current CUDA stream (primarily for runtime tests)."""
    _require_cuda()
    _cuda_core._sleep(int(cycles))


def manual_seed(seed: int) -> None:
    _require_cuda()
    _cuda_core.manual_seed(int(seed))


def manual_seed_all(seed: int) -> None:
    _require_cuda()
    _cuda_core.manual_seed_all(int(seed))


def cudart():
    """TensorPlay does not currently expose a ctypes CUDA runtime wrapper."""
    return None


__all__ = [
    "Event", "Stream", "current_device", "current_stream", "default_stream",
    "device", "device_count", "device_of", "empty_cache", "get_device_capability",
    "get_device_name", "get_device_properties", "get_stream_priority_range",
    "init", "is_available", "is_initialized", "manual_seed", "manual_seed_all",
    "max_memory_allocated", "max_memory_reserved", "memory_allocated",
    "memory_reserved", "memory_stats", "memory_summary", "reset_peak_memory_stats",
    "set_device", "set_stream", "stream", "synchronize",
]
