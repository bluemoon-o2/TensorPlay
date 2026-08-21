"""TensorPlay model serialization.

TensorPlay's native serialization format is MEGA.  The implementation is
kept in :mod:`megatensors`: TensorPlay supplies the framework adapter and
``megatensors`` owns the canonical MEGA header, payload layout, and loader.

Only ``.mega`` files (and MEGA shard indexes when loading) are accepted here.
Legacy containers and third-party checkpoint formats are intentionally not
part of this API.
"""

from __future__ import annotations

import importlib
import os
from collections import OrderedDict
from typing import Any, Mapping

import tensorplay as tp


MEGA_EXTENSION = ".mega"
MEGA_INDEX_SUFFIX = ".mega.index.json"
DEFAULT_ALIGNMENT = 4096


def _require_mega_path(filename: str, *, allow_index: bool = False) -> str:
    valid = filename.endswith(MEGA_EXTENSION) or (
        allow_index and filename.endswith(MEGA_INDEX_SUFFIX)
    )
    if not valid:
        raise ValueError(
            "TensorPlay serialization only supports MEGA files with a '.mega' "
            f"extension (got {filename!r})"
        )
    return filename


def _import_megatensors():
    """Import the installed MEGA backend and verify TensorPlay support."""

    def is_compatible(module: Any) -> bool:
        return callable(getattr(module, "write_tensorplay_file", None))

    try:
        module = importlib.import_module("megatensors")
    except ImportError as error:
        raise ImportError(
            "TensorPlay MEGA serialization requires the megatensors package "
            "with TensorPlay support; install megatensors>=0.0.5."
        ) from error
    if not is_compatible(module):
        raise ImportError(
            "TensorPlay MEGA serialization requires a megatensors package "
            "with the TensorPlay adapter"
        )
    return module


def _device_string(map_location: Any) -> str:
    if map_location is None:
        return "cpu"
    if isinstance(map_location, str):
        return map_location
    if isinstance(map_location, tp.Device):
        return str(map_location)
    if isinstance(map_location, Mapping):
        value = map_location.get("cpu", "cpu")
        return str(value)
    if callable(map_location):
        value = map_location("cpu")
        return str(value) if value is not None else "cpu"
    raise TypeError(
        "map_location must be a device string, TensorPlay Device, mapping, or callable"
    )


def _container_from_metadata(metadata: Mapping[str, Any]) -> str:
    container = str(metadata.get("tensorplay.container", "dict"))
    if container not in {"dict", "tuple", "list", "tensor"}:
        raise ValueError(f"Unsupported TensorPlay MEGA container type: {container!r}")
    return container


def _load_metadata(artifact: Any) -> Mapping[str, Any]:
    metadata_by_file = artifact.metadata()
    if not metadata_by_file:
        return {}
    return next(iter(metadata_by_file.values()))


def save(
    obj: Any,
    f: str | os.PathLike[str],
    *,
    metadata: Mapping[str, Any] | None = None,
    alignment: int = DEFAULT_ALIGNMENT,
) -> None:
    """Save a TensorPlay tensor container as a MEGA ``.mega`` artifact.

    ``obj`` may be a tensor mapping (the usual ``state_dict`` form), tuple, or
    list of tensors.  Arbitrary Python objects are deliberately unsupported:
    MEGA is a tensor/weights format, not a pickle container.
    """

    filename = _require_mega_path(os.fspath(f))
    megatensors = _import_megatensors()
    megatensors.write_tensorplay_file(
        filename,
        obj,
        metadata=metadata,
        alignment=int(alignment),
    )


def load(
    f: str | os.PathLike[str],
    map_location: Any = None,
    **kwargs: Any,
) -> Any:
    """Load a TensorPlay tensor container from a MEGA artifact.

    MEGA shard indexes are accepted for sharded artifacts.  No legacy
    container or third-party checkpoint loading is exposed through TensorPlay.
    """

    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported TensorPlay MEGA load arguments: {unknown}")
    filename = _require_mega_path(os.fspath(f), allow_index=True)
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No such file or directory: '{filename}'")

    megatensors = _import_megatensors()
    device = _device_string(map_location)
    with megatensors.mega_open(
        filename,
        framework="tensorplay",
        device=device,
        nogds=True,
    ) as artifact:
        state = OrderedDict(
            (name, artifact.get_tensor(name).clone()) for name in artifact.keys()
        )
        container = _container_from_metadata(_load_metadata(artifact))

    if container == "tuple":
        return tuple(state.values())
    if container == "list":
        return list(state.values())
    if container == "tensor":
        if len(state) != 1:
            raise ValueError("TensorPlay MEGA tensor container must contain one tensor")
        return next(iter(state.values()))
    return state
