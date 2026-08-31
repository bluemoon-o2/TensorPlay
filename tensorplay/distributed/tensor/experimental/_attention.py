"""Context-parallel attention buffer management."""

from __future__ import annotations

import contextlib
from enum import Enum
from typing import Any, Iterator

from .._api import DTensor
from ..placement_types import Shard, Replicate

__all__ = ["context_parallel", "context_parallel_unshard", "set_rotate_method"]


class _CausalBehavior(str, Enum):
    UP = "up"
    DOWN = "down"


class _RotateMethod(str, Enum):
    ALL_GATHER = "allgather"
    ALL_TO_ALL = "alltoall"


_rotate_method = _RotateMethod.ALL_GATHER


def set_rotate_method(rotate_method: str) -> None:
    global _rotate_method
    try:
        _rotate_method = _RotateMethod(rotate_method)
    except ValueError as exc:
        raise ValueError("rotate_method must be 'allgather' or 'alltoall'") from exc


def _shard_buffer(buffer: Any, mesh: Any, dim: int) -> Any:
    if isinstance(buffer, DTensor):
        return buffer.redistribute(placements=[Shard(dim)])
    return DTensor.from_local(buffer, mesh, [Replicate()]).redistribute(placements=[Shard(dim)]).to_local()


@contextlib.contextmanager
def context_parallel(mesh: Any, *, buffers: list[Any] | None = None, buffer_seq_dims: list[int] | None = None, no_restore_buffers: set[Any] | None = None) -> Iterator[None]:
    buffers = list(buffers or [])
    dims = list(buffer_seq_dims or [])
    if len(buffers) != len(dims):
        raise ValueError("buffers and buffer_seq_dims must have equal length")
    no_restore = no_restore_buffers or set()
    if not all(any(buffer is item for item in buffers) for buffer in no_restore):
        raise ValueError("no_restore_buffers must be a subset of buffers")
    originals = [None if any(buffer is item for item in no_restore) else buffer.clone() for buffer in buffers]
    for buffer, dim in zip(buffers, dims):
        shard = _shard_buffer(buffer, mesh, dim)
        if hasattr(buffer, "resize_"):
            buffer.resize_(shard.shape)
            buffer.copy_(shard)
    try:
        yield
    finally:
        for buffer, original in zip(buffers, originals):
            if original is not None and hasattr(buffer, "resize_"):
                buffer.resize_(original.shape)
                buffer.copy_(original)


def context_parallel_unshard(mesh: Any, buffers: list[Any], seq_dims: list[int], load_balancer: Any = None) -> list[Any]:
    del load_balancer
    result = []
    for buffer, dim in zip(buffers, seq_dims):
        if isinstance(buffer, DTensor):
            result.append(buffer.redistribute(placements=[Replicate()]).to_local())
        else:
            result.append(buffer)
    return result
