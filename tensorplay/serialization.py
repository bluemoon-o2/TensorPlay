"""TensorPlay model serialization.

The native format is MEGA (``.mega``), implemented on top of :mod:`megatensors`.
TensorPlay adds nested-container support, storage-sharing deduplication,
checksums, endianness records, zero-copy ``mmap`` loading, and full
``map_location`` semantics on that foundation.

Interoperability formats are accepted transparently by extension or content,
including ``.safetensors`` files. Saving to a file-like object (e.g.

All loaders are allowlist-based (weights-only): loading never executes code
embedded in a checkpoint.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import warnings
from collections import OrderedDict
from typing import Any, Mapping

import tensorplay as tp

from ._serialization_torch import (
    describe_safetensors_file,
    describe_torch_file,
    read_safetensors_file,
    read_torch_file,
    write_safetensors_file,
    write_torch_file,
)

__all__ = [
    "save",
    "load",
    "inspect_checkpoint",
    "MEGA_EXTENSION",
    "MEGA_INDEX_SUFFIX",
    "DEFAULT_ALIGNMENT",
]

MEGA_EXTENSION = ".mega"
MEGA_INDEX_SUFFIX = ".mega.index.json"
DEFAULT_ALIGNMENT = 4096

_MEGA_MAGIC = b"MEGA"

# MEGA metadata type tags used by the extension.
_MEGA_META_UINT32 = 4
_MEGA_META_INT32 = 5
_MEGA_META_UINT64 = 6
_MEGA_META_INT64 = 7
_MEGA_META_FLOAT32 = 8
_MEGA_META_FLOAT64 = 9
_MEGA_META_BOOL = 10
_MEGA_META_STRING = 11
_MEGA_META_ARRAY = 12

_CHECKSUM_TYPES = {"none": None, "crc32": 1, "sha256": 2}


# ---------------------------------------------------------------------------
# helpers shared with the interop layer
# ---------------------------------------------------------------------------


class _StorageStub:
    """Minimal stand-in handed to callable map_location implementations."""

    __slots__ = ("device", "nbytes")

    def __init__(self, location: str, nbytes: int = 0):
        self.device = location
        self.nbytes = nbytes


def _validate_device_spec(spec: str) -> str:
    if spec.startswith("cuda"):
        if not hasattr(tp, "cuda") or not tp.cuda.is_available():
            raise RuntimeError(
                f"Attempting to deserialize onto {spec} but TensorPlay CUDA "
                "support is not available; use map_location='cpu'."
            )
    elif spec.startswith(("gpu", "npu", "xpu", "mps")):
        raise RuntimeError(
            f"TensorPlay serialization does not support device {spec!r}; "
            "supported targets are 'cpu' and 'cuda[:index]'."
        )
    return spec


def resolve_map_location(map_location: Any, location: str):
    """Resolve ``map_location`` for one storage saved at ``location``.

    Returns ``None`` (keep the saved location), a device string such as
    ``"cuda:0"``, or a TensorPlay tensor whose contents should receive the
    storage-returning callables).
    """

    if map_location is None:
        return None
    if isinstance(map_location, str):
        return _validate_device_spec(map_location)
    if isinstance(map_location, tp.Device):
        return _validate_device_spec(str(map_location))
    if isinstance(map_location, Mapping):
        target = map_location.get(location, location)
        return _validate_device_spec(str(target))
    if callable(map_location):
        result = map_location(_StorageStub(location), location)
        if result is None:
            return None
        if isinstance(result, tp.Tensor):
            return result
        if isinstance(result, tp.Device):
            return _validate_device_spec(str(result))
        return _validate_device_spec(str(result))
    raise TypeError(
        "map_location must be None, a device string, a TensorPlay Device, a "
        "mapping, or a callable"
    )


def _apply_location(flat, resolved):
    """Move (or copy into a user-provided tensor) a freshly loaded flat tensor."""

    if resolved is None:
        return flat
    if isinstance(resolved, tp.Tensor):
        if resolved.numel() != flat.numel():
            raise ValueError(
                f"map_location returned a tensor with {resolved.numel()} elements "
                f"but the checkpoint storage holds {flat.numel()}"
            )
        resolved.copy_(flat.reshape(resolved.shape))
        return resolved
    return _move_to(flat, resolved)


def _move_to(tensor, spec: str):
    if spec == "cpu":
        return tensor
    return tensor.to(_parse_device(spec))


def _parse_device(spec: str):
    if spec == "cpu":
        return tp.Device(tp.DeviceType.CPU)
    if ":" in spec:
        kind, index = spec.split(":", 1)
        return tp.Device(getattr(tp.DeviceType, kind.upper()), int(index))
    return tp.Device(getattr(tp.DeviceType, spec.upper()))


def _byteswap_tensor(tensor):
    """Return a byte-swapped copy of a contiguous CPU tensor."""

    from ._serialization_torch import (
        _contiguous_stride,
        _dtype_name_of,
        _tensor_from_flat_bytes,
    )

    import numpy as np

    if not tensor.is_contiguous():
        raise ValueError("byte-swapping requires a contiguous tensor")
    if tensor.numel() == 0:
        return tensor
    dtype_name = _dtype_name_of(tensor)
    if dtype_name == "bfloat16":
        array = tensor.view(_tp_uint16()).numpy()
    else:
        array = tensor.numpy()
    data = array.byteswap().tobytes()
    flat = _tensor_from_flat_bytes(data, dtype_name)
    shape = [int(dim) for dim in tensor.shape]
    stride = _contiguous_stride(shape)
    if stride == _contig_stride(shape):
        return flat.reshape(shape) if shape else flat.reshape([])
    return flat.as_strided(shape, stride, 0)


def _tp_uint16():
    dtype = getattr(tp, "uint16", None)
    if dtype is None:
        raise NotImplementedError("TensorPlay lacks uint16 required for bfloat16 byteswap")
    return dtype


def _contig_stride(size):
    stride = [0] * len(size)
    running = 1
    for index in range(len(size) - 1, -1, -1):
        stride[index] = running
        running *= size[index]
    return stride


# ---------------------------------------------------------------------------
# MEGA header parsing (pure Python; follows the local extension layout)
# ---------------------------------------------------------------------------


def _read_string(buf: memoryview, pos: tuple):
    (length,) = struct.unpack_from("<Q", buf, pos[0])
    pos[0] += 8
    value = bytes(buf[pos[0]:pos[0] + length]).decode("utf-8")
    pos[0] += length
    return value


def _read_compact_string(buf: memoryview, pos: tuple):
    (length,) = struct.unpack_from("<I", buf, pos[0])
    pos[0] += 4
    value = bytes(buf[pos[0]:pos[0] + length]).decode("utf-8")
    pos[0] += length
    return value


def _read_meta_value(buf: memoryview, pos: tuple):
    (tag,) = struct.unpack_from("<I", buf, pos[0])
    pos[0] += 4
    if tag == _MEGA_META_STRING:
        return _read_string(buf, pos)
    if tag == _MEGA_META_BOOL:
        value = bool(buf[pos[0]])
        pos[0] += 1
        return value
    formats = {
        _MEGA_META_UINT32: "<I", _MEGA_META_INT32: "<i",
        _MEGA_META_UINT64: "<Q", _MEGA_META_INT64: "<q",
        _MEGA_META_FLOAT32: "<f", _MEGA_META_FLOAT64: "<d",
    }
    if tag in formats:
        (value,) = struct.unpack_from(formats[tag], buf, pos[0])
        pos[0] += struct.calcsize(formats[tag])
        return value
    fixed = {0: 1, 1: 1, 2: 2, 3: 2}
    if tag in fixed:
        width = fixed[tag]
        (value,) = struct.unpack_from("<I", buf, pos[0])
        pos[0] += width
        return value
    if tag == _MEGA_META_ARRAY:
        (elem_tag,) = struct.unpack_from("<I", buf, pos[0])
        pos[0] += 4
        (count,) = struct.unpack_from("<Q", buf, pos[0])
        pos[0] += 8
        items = []
        for _ in range(count):
            items.append(_read_meta_value_of_tag(buf, pos, elem_tag))
        return items
    raise ValueError(f"unknown MEGA metadata tag {tag}")


def _read_meta_value_of_tag(buf: memoryview, pos: tuple, tag: int):
    if tag == _MEGA_META_STRING:
        return _read_string(buf, pos)
    if tag == _MEGA_META_BOOL:
        value = bool(buf[pos[0]])
        pos[0] += 1
        return value
    formats = {
        _MEGA_META_UINT32: "<I", _MEGA_META_INT32: "<i",
        _MEGA_META_UINT64: "<Q", _MEGA_META_INT64: "<q",
        _MEGA_META_FLOAT32: "<f", _MEGA_META_FLOAT64: "<d",
    }
    if tag in formats:
        (value,) = struct.unpack_from(formats[tag], buf, pos[0])
        pos[0] += struct.calcsize(formats[tag])
        return value
    fixed = {0: 1, 1: 1, 2: 2, 3: 2}
    if tag in fixed:
        (value,) = struct.unpack_from("<I", buf, pos[0])
        pos[0] += fixed[tag]
        return value
    raise ValueError(f"unknown MEGA metadata element tag {tag}")


def parse_mega_header(path: str) -> dict:
    """Parse a MEGA file header without loading any payload bytes."""

    with open(path, "rb") as handle:
        prefix = handle.read(24)
        if len(prefix) < 24 or prefix[:4] != _MEGA_MAGIC:
            raise ValueError(f"{path}: not a MEGA file (bad magic)")
        (version,) = struct.unpack_from("<I", prefix, 4)
        num_tensors, num_metadata = struct.unpack_from("<QQ", prefix, 8)
        header_rest = handle.read(64 * 1024 * 1024)
    if len(prefix) + len(header_rest) < 24 + 8 * num_tensors + 16 * num_metadata:
        raise ValueError(f"{path}: MEGA header truncated")
    buf = memoryview(prefix + header_rest)
    pos = [24]

    metadata = {}
    for _ in range(num_metadata):
        key = _read_string(buf, pos)
        metadata[key] = _read_meta_value(buf, pos)

    tensors = OrderedDict()
    for _ in range(num_tensors):
        name = _read_compact_string(buf, pos)
        (dir_flags,) = struct.unpack_from("<I", buf, pos)
        pos[0] += 4
        (ndim,) = struct.unpack_from("<I", buf, pos)
        pos[0] += 4
        dims = []
        for _dim in range(ndim):
            (dim,) = struct.unpack_from("<Q", buf, pos)
            pos[0] += 8
            dims.append(dim)
        dims.reverse()  # written most-significant dim last
        dtype = _read_compact_string(buf, pos)
        (payload_offset, logical_nbytes) = struct.unpack_from("<QQ", buf, pos)
        pos[0] += 16
        stored_nbytes = logical_nbytes
        checksum_type = 0
        if dir_flags & 0x1:
            _read_compact_string(buf, pos)  # storage_format
        if dir_flags & 0x2:
            (stored_nbytes,) = struct.unpack_from("<Q", buf, pos)
            pos[0] += 8
        if dir_flags & 0x4:
            pos[0] += 4  # tensor_flags
        if dir_flags & 0x8:
            pos[0] += 4  # compression_codec
        if dir_flags & 0x10:
            pos[0] += 4  # shuffle_elem_size
        if dir_flags & 0x20:
            (checksum_type,) = struct.unpack_from("<I", buf, pos)
            pos[0] += 4 + 32
        itemsize = _MEGA_DTYPE_SIZES.get(dtype)
        if itemsize is None:
            raise NotImplementedError(f"unsupported MEGA dtype {dtype!r}")
        tensors[name] = {
            "shape": dims,
            "dtype": dtype,
            "itemsize": itemsize,
            "payload_offset": payload_offset,
            "logical_nbytes": logical_nbytes,
            "stored_nbytes": stored_nbytes,
            "checksum_type": checksum_type,
        }

    alignment = int(metadata.get("general.alignment", 4096) or 4096)
    header_length = ((pos[0] + alignment - 1) // alignment) * alignment
    return {
        "version": version,
        "metadata": metadata,
        "tensors": tensors,
        "raw_header_size": pos[0],
        "header_length": header_length,
    }


_MEGA_DTYPE_SIZES = {
    "BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2, "I32": 4, "U32": 4,
    "I64": 8, "U64": 8, "F16": 2, "BF16": 2, "F32": 4, "F64": 8,
    "F8_E5M2": 1, "F8_E4M3": 1, "F8_E8M0": 1,
}


# ---------------------------------------------------------------------------
# container flattening (tree <-> flat tensor table)
# ---------------------------------------------------------------------------

_PRIMITIVES = (bool, int, float, str)


def _is_tensor(value: Any) -> bool:
    parameter_cls = getattr(tp.nn, "Parameter", None)
    return isinstance(value, tp.Tensor) or (
        parameter_cls is not None and isinstance(value, parameter_cls)
    )


def _flatten_tree(obj: Any) -> tuple:
    """Flatten ``obj`` into ``(flat_table, container_kind, layout_or_None)``."""

    if _is_tensor(obj):
        return OrderedDict([("0", obj)]), "tensor", None

    def assign(layout_leaf, path, flat, names_used):
        name = ".".join(str(part) for part in path) or "0"
        base = name
        counter = 1
        while name in names_used:
            name = f"{base}#{counter}"
            counter += 1
        names_used.add(name)
        flat[name] = layout_leaf
        return {"__tensor__": name}

    flat: OrderedDict = OrderedDict()
    names_used: set = set()

    def encode(node, path):
        if _is_tensor(node):
            return assign(node, path, flat, names_used)
        if isinstance(node, _PRIMITIVES) or node is None:
            return node
        if isinstance(node, dict):
            encoded = {}
            for key, value in node.items():
                if not isinstance(key, str):
                    raise TypeError(
                        f"dict keys must be str for MEGA serialization (got {type(key).__name__} at "
                        f"{'.'.join(map(str, path)) or '<root>'})"
                    )
                encoded[key] = encode(value, path + [key])
            return encoded
        if isinstance(node, list):
            return [encode(value, path + [index]) for index, value in enumerate(node)]
        if isinstance(node, tuple):
            return {
                "__tuple__": [
                    encode(value, path + [index]) for index, value in enumerate(node)
                ]
            }
        raise TypeError(
            f"unsupported leaf of type {type(node).__name__} at "
            f"{'.'.join(map(str, path)) or '<root>'}: MEGA stores tensors and "
            "JSON primitives (None/bool/int/float/str); use '.pt' for arbitrary objects"
        )

    layout = encode(obj, [])

    if isinstance(obj, Mapping) and all(_is_tensor(v) for v in obj.values()):
        return OrderedDict((key, obj[key]) for key in obj), "dict", None
    if isinstance(obj, tuple) and all(_is_tensor(v) for v in obj):
        return OrderedDict(
            (str(index), value) for index, value in enumerate(obj)
        ), "tuple", None
    if isinstance(obj, list) and all(_is_tensor(v) for v in obj):
        return OrderedDict(
            (str(index), value) for index, value in enumerate(obj)
        ), "list", None
    return flat, "tree", layout


def _rebuild_tree(layout, tensors):
    if isinstance(layout, dict):
        if "__tensor__" in layout and len(layout) == 1:
            return tensors[layout["__tensor__"]]
        if "__tuple__" in layout and len(layout) == 1:
            return tuple(_rebuild_tree(item, tensors) for item in layout["__tuple__"])
        return {key: _rebuild_tree(value, tensors) for key, value in layout.items()}
    if isinstance(layout, list):
        return [_rebuild_tree(item, tensors) for item in layout]
    return layout


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def _require_megatensors():
    def is_compatible(module: Any) -> bool:
        return callable(getattr(module, "write_tensorplay_file", None))

    try:
        module = __import__("megatensors")
    except ImportError as error:
        raise ImportError(
            "TensorPlay MEGA serialization requires the megatensors package "
            "(megatensors>=0.0.5)."
        ) from error
    if not is_compatible(module):
        raise ImportError(
            "TensorPlay MEGA serialization requires a megatensors package with "
            "the TensorPlay adapter"
        )
    return module


def _is_path_like(f: Any) -> bool:
    return isinstance(f, (str, os.PathLike))


def _sniff_format(head: bytes) -> str | None:
    if head[:4] == b"PK\x03\x04":
        return "torch_zip"
    if head[:4] == _MEGA_MAGIC:
        return "mega"
    if len(head) >= 9:
        (header_length,) = struct.unpack_from("<Q", head, 0)
        if 10 <= header_length <= 100_000_000 and head[8:9] == b"{":
            return "safetensors"
    if len(head) > 257 and head[257:262] == b"ustar":
        return "torch_tar"
    return None


def _probe_torch_stream(path: str) -> bool:
    """Check for the legacy magic-number prefix without executing pickle code.

    Uses the allowlist-restricted unpickler with empty resolvers, so probing
    an untrusted file can only ever read a leading integer, never run GLOBAL
    reductions.
    """

    from ._serialization_torch import _WeightsOnlyUnpickler

    try:
        with open(path, "rb") as handle:
            first = _WeightsOnlyUnpickler(
                handle,
                persistent_load=lambda saved_id: None,
                resolve_global=lambda module, name: None,
            ).load()
        return first == 0x1950A86A20F9469CFC6C
    except Exception:
        return False


def save(
    obj: Any,
    f: str | os.PathLike[str],
    *,
    metadata: Mapping[str, Any] | None = None,
    alignment: int = DEFAULT_ALIGNMENT,
    checksum: str = "crc32",
    pickle_protocol: int = 2,
) -> None:
    """Save ``obj`` to disk.

    Formats are selected by extension:

    - ``.mega`` (default): native MEGA artifact.  Supports nested tensor
      containers plus JSON-primitive leaves, shared-storage deduplication, an
      ``alignment``, per-region ``checksum`` (``"crc32"``, ``"sha256"``, or
      ``"none"``) and free-form ``metadata`` (strings, numbers, bools, lists).
    - ``.safetensors``: flat name->tensor mapping.

    """

    if not _is_path_like(f):
        write_torch_file(f, obj, pickle_protocol=pickle_protocol)
        return

    filename = os.fspath(f)
    lower = filename.lower()
    if lower.endswith(MEGA_INDEX_SUFFIX):
        raise ValueError(
            "cannot save directly to a shard index; save the sharded artifacts "
            "with megatensors.convert_model"
        )
    if lower.endswith(MEGA_EXTENSION):
        _save_mega(filename, obj, metadata=metadata, alignment=alignment,
                   checksum=checksum)
        return
    if lower.endswith(".safetensors"):
        with open(filename, "wb") as handle:
            write_safetensors_file(handle, obj, metadata=metadata)
        return
    if lower.endswith((".pt", ".pth")):
        with open(filename, "wb") as handle:
            write_torch_file(handle, obj, pickle_protocol=pickle_protocol)
        return
    raise ValueError(
        f"unsupported checkpoint extension: {filename!r}. Supported: "
        "'.mega', '.safetensors', '.pt', '.pth'"
    )


def _save_mega(filename: str, obj: Any, *, metadata, alignment: int, checksum: str):
    if checksum not in _CHECKSUM_TYPES:
        raise ValueError(
            f"checksum must be one of {sorted(_CHECKSUM_TYPES)} (got {checksum!r})"
        )
    megatensors = _require_megatensors()
    flat, container, layout = _flatten_tree(obj)
    merged_metadata = dict(metadata or {})
    megatensors.write_tensorplay_file(
        filename,
        flat,
        metadata=merged_metadata,
        alignment=int(alignment),
        checksum=checksum,
        container=container,
        layout=layout,
    )


def load(
    f: str | os.PathLike[str],
    map_location: Any = None,
    *,
    mmap: bool = False,
    weights_only: bool = True,
) -> Any:
    """

    Format is detected by extension or content sniffing: ``.mega`` (and MEGA
    zip, magic-number stream, and tar layouts).

    Args:
        f: path or file-like object (seekable binary).
        map_location: controls where tensors land.  Accepts a device string /
            :class:`tensorplay.Device` (all tensors), a mapping of saved
            location -> target, or a callable invoked per storage as
            ``fn(storage_stub, location)`` returning ``None``, a device, or a
            TensorPlay tensor receiving the data.
        mmap: if true, MEGA and safetensors tensors are zero-copy views over a
            private (copy-on-write) mapping of the file; the returned
            containers keep the mapping alive, and bytes are paged in on first
            load eagerly.
        weights_only: accepted for API familiarity; every TensorPlay loader is
            weights-only by construction.  Passing ``False`` warns.
    """

    if not weights_only:
        warnings.warn(
            "TensorPlay loaders are weights-only by construction; "
            "weights_only=False has no additional effect.",
            stacklevel=2,
        )

    if not _is_path_like(f):
        return _load_stream(f, map_location, mmap=mmap)

    filename = os.fspath(f)
    if filename.endswith(MEGA_INDEX_SUFFIX):
        return _load_mega(filename, map_location, mmap=False)
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No such file or directory: '{filename}'")

    with open(filename, "rb") as handle:
        head = handle.read(512)
    kind = _sniff_format(head)
    if kind == "mega":
        return _load_mega(filename, map_location, mmap=mmap)
    if kind == "torch_zip":
        with open(filename, "rb") as handle:
            return read_torch_file(handle, map_location=map_location)
    if kind == "safetensors":
        with open(filename, "rb") as handle:
            return read_safetensors_file(handle, map_location=map_location, mmap=mmap)
    if kind == "torch_tar":
        with open(filename, "rb") as handle:
            return read_torch_file(handle, map_location=map_location)
    if _probe_torch_stream(filename):
        with open(filename, "rb") as handle:
            return read_torch_file(handle, map_location=map_location)

    lower = filename.lower()
    if lower.endswith(MEGA_EXTENSION):
        raise ValueError(f"{filename}: not a valid MEGA file (bad magic)")
    if lower.endswith(".safetensors"):
        raise ValueError(f"{filename}: not a valid safetensors file")
    if lower.endswith((".pt", ".pth")):
        raise ValueError(f"{filename}: not a recognized legacy checkpoint")
    raise ValueError(
        f"unrecognized checkpoint: {filename!r}. Supported: '.mega', "
    )


def _load_stream(fileobj, map_location, *, mmap: bool = False) -> Any:
    position = fileobj.tell()
    head = fileobj.read(512)
    fileobj.seek(position)
    kind = _sniff_format(head)
    if kind == "torch_zip":
        return read_torch_file(fileobj, map_location=map_location)
    if kind == "safetensors":
        return read_safetensors_file(fileobj, map_location=map_location, mmap=mmap)
    if kind == "mega":
        # MEGA loading goes through the C++ file loader; spool to disk first.
        import shutil
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=MEGA_EXTENSION, delete=False) as spool:
            shutil.copyfileobj(fileobj, spool)
            spool_path = spool.name
        try:
            return _load_mega(spool_path, map_location, mmap=False)
        finally:
            os.unlink(spool_path)
    return read_torch_file(fileobj, map_location=map_location)


class _ArtifactAware(OrderedDict):
    """OrderedDict that keeps a MEGA mapping alive for mmap-loaded views."""

    _keepalive: Any = None


class _ArtifactList(list):
    _keepalive: Any = None


def _load_mega(filename: str, map_location, *, mmap: bool) -> Any:
    megatensors = _require_megatensors()
    header = parse_mega_header(filename)
    meta = header["metadata"]
    infos = header["tensors"]

    saved_byteorder = str(meta.get("tensorplay.byteorder", sys.byteorder))
    swap = saved_byteorder != sys.byteorder
    container_kind = str(meta.get("tensorplay.container", "dict"))
    devices: dict = {}
    if "tensorplay.tensor_devices" in meta:
        try:
            devices = json.loads(meta["tensorplay.tensor_devices"])
        except json.JSONDecodeError:
            devices = {}
    layout = None
    if "tensorplay.layout" in meta:
        layout = json.loads(meta["tensorplay.layout"])

    if isinstance(map_location, (str, tp.Device)):
        open_device = str(map_location if isinstance(map_location, str) else map_location)
        open_device = _validate_device_spec(open_device)
    else:
        open_device = "cpu"

    artifact = megatensors.mega_open(
        filename,
        framework="tensorplay",
        device=open_device,
        nogds=True,
    )

    keepalive = artifact if mmap else None

    def load_base(name: str):
        tensor = artifact.get_tensor(name)
        if getattr(tensor, "is_contiguous", lambda: True)() is False:
            tensor = tensor.contiguous()
        if swap:
            tensor = _byteswap_tensor(tensor)
        saved_location = devices.get(name, open_device)
        resolved = resolve_map_location(map_location, saved_location)
        return _apply_location(tensor, resolved)

    try:
        # Group descriptors sharing a payload region: siblings become
        # as_strided aliases of the first-loaded tensor.
        region_owner: dict = {}
        alias_specs: list = []
        for name, info in infos.items():
            region = (info["payload_offset"], info["logical_nbytes"])
            owner = region_owner.get(region)
            if owner is None:
                region_owner[region] = name
            else:
                alias_specs.append((name, owner, info))

        state = {}
        for name in infos:
            if name not in state:
                state[name] = load_base(name)
        for name, owner, info in alias_specs:
            base = state[owner]
            shape = [int(dim) for dim in info["shape"]]
            state[name] = base.as_strided(shape, _contig_stride(shape), 0)

        ordered = OrderedDict((name, state[name]) for name in infos)

        if container_kind == "tree" and layout is not None:
            result = _rebuild_tree(layout, ordered)
        elif container_kind == "tuple":
            result = tuple(ordered.values())
        elif container_kind == "list":
            result = list(ordered.values())
        elif container_kind == "tensor":
            if len(ordered) != 1:
                raise ValueError("MEGA tensor container must contain exactly one tensor")
            result = next(iter(ordered.values()))
        else:
            result = ordered

        if keepalive is not None:
            result = _attach_keepalive(result, keepalive)
        else:
            artifact.close()
        return result
    except BaseException:
        if keepalive is None:
            try:
                artifact.close()
            except Exception:
                pass
        raise


def _attach_keepalive(result, artifact):
    if isinstance(result, OrderedDict):
        holder = _ArtifactAware(result)
        holder._keepalive = artifact
        return holder
    if isinstance(result, list):
        holder = _ArtifactList(result)
        holder._keepalive = artifact
        return holder
    try:
        result._mega_keepalive = artifact
        return result
    except AttributeError:
        warnings.warn(
            "mmap=True cannot keep the MEGA mapping alive for this container; "
            "falling back to an eager copy.",
            stacklevel=2,
        )
        return _deep_copy_result(result)


def _deep_copy_result(result):
    if isinstance(result, OrderedDict):
        return OrderedDict((key, value.clone()) for key, value in result.items())
    if isinstance(result, list):
        return [value.clone() for value in result]
    if isinstance(result, tuple):
        return tuple(value.clone() for value in result)
    return result.clone()


# ---------------------------------------------------------------------------
# inspection
# ---------------------------------------------------------------------------


def inspect_checkpoint(f: str | os.PathLike[str], *, verify_checksums: bool = False) -> dict:
    """Summarize a checkpoint without loading tensor payloads.

    Returns a dict describing the container format, metadata, and per-tensor
    shape/dtype/extent information.  With ``verify_checksums=True`` MEGA
    payloads are streamed through their stored CRC32/SHA256 digests (the MEGA
    loader also verifies automatically whenever tensors are loaded).
    """

    filename = os.fspath(f)
    if filename.endswith(MEGA_INDEX_SUFFIX):
        return {"format": "mega_shard_index"}
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No such file or directory: '{filename}'")
    with open(filename, "rb") as handle:
        head = handle.read(512)
    kind = _sniff_format(head)

    if kind == "mega":
        header = parse_mega_header(filename)
        tensors = {}
        for name, info in header["tensors"].items():
            tensors[name] = {
                "shape": info["shape"],
                "dtype": info["dtype"],
                "nbytes": info["logical_nbytes"],
                "payload_offset": info["payload_offset"],
                "checksum": {0: None, 1: "crc32", 2: "sha256"}.get(info["checksum_type"]),
            }
        result = {
            "format": "mega",
            "version": header["version"],
            "metadata": header["metadata"],
            "tensors": tensors,
            "header_length": header["header_length"],
            "file_size": os.path.getsize(filename),
        }
        if verify_checksums:
            # The MEGA loader verifies every stored digest while materializing
            # a tensor; loading each name exercises that path.
            megatensors = _require_megatensors()
            with megatensors.mega_open(
                filename, framework="tensorplay", device="cpu", nogds=True
            ) as artifact:
                for name in header["tensors"]:
                    artifact.get_tensor(name)
            result["checksums_verified"] = True
        return result
    if kind == "torch_zip":
        with open(filename, "rb") as handle:
            return describe_torch_file(handle)
    if kind == "safetensors":
        with open(filename, "rb") as handle:
            return describe_safetensors_file(handle)
    if kind == "torch_tar":
        return {"format": "torch_legacy_tar"}
    if _probe_torch_stream(filename):
        return {"format": "torch_stream"}
    raise ValueError(f"unrecognized checkpoint: {filename!r}")
