"""Checkpoint serialization helpers for legacy stream formats.

magic-number stream format saved with ``_use_new_zipfile_serialization=False``

The unpickler is allowlist-based (weights-only semantics): only tensors,
globals are accepted, so loading a checkpoint never executes arbitrary code.

Performance note: this layer orchestrates; bytes never flow through the
interpreter.  Pickle payloads hold metadata only (shapes/strides/keys), bulk
data moves via ``np.frombuffer`` views and the C++ ``Tensor._from_bytes``
"""

from __future__ import annotations

import copyreg
import io
import json
import pickle
import struct
import sys
import tarfile
import zipfile
from collections import OrderedDict
from types import FunctionType
from typing import Any, BinaryIO, Mapping

import tensorplay as tp

__all__ = [
    "read_torch_file",
    "write_torch_file",
    "describe_torch_file",
    "read_safetensors_file",
    "write_safetensors_file",
    "describe_safetensors_file",
]


TORCH_MAGIC_NUMBER = 0x1950A86A20F9469CFC6C
TORCH_PROTOCOL_VERSION = 1001


# ---------------------------------------------------------------------------
# dtype plumbing
# ---------------------------------------------------------------------------

_TORCH_STORAGE_DTYPES = {
    "DoubleStorage": "float64",
    "FloatStorage": "float32",
    "HalfStorage": "float16",
    "BFloat16Storage": "bfloat16",
    "LongStorage": "int64",
    "IntStorage": "int32",
    "ShortStorage": "int16",
    "CharStorage": "int8",
    "ByteStorage": "uint8",
    "BoolStorage": "bool",
    "ComplexDoubleStorage": "complex128",
    "ComplexFloatStorage": "complex64",
    "UntypedStorage": "uint8",
}

_STORAGE_NAMES_BY_DTYPE = {
    "float64": "DoubleStorage",
    "float32": "FloatStorage",
    "float16": "HalfStorage",
    "bfloat16": "BFloat16Storage",
    "int64": "LongStorage",
    "int32": "IntStorage",
    "int16": "ShortStorage",
    "int8": "CharStorage",
    "uint8": "ByteStorage",
    "bool": "BoolStorage",
    "complex128": "ComplexDoubleStorage",
    "complex64": "ComplexFloatStorage",
}

_ITEMSIZE = {
    "bool": 1, "uint8": 1, "int8": 1, "int16": 2, "uint16": 2, "int32": 4,
    "uint32": 4, "int64": 8, "uint64": 8, "float16": 2, "bfloat16": 2,
    "float32": 4, "float64": 8, "complex64": 8, "complex128": 16,
}

_NUMPY_DTYPES = {
    "bool": "?", "uint8": "u1", "int8": "i1", "int16": "i2", "int32": "i4",
    "int64": "i8", "uint32": "u4", "uint64": "u8", "float16": "f2",
    "float32": "f4", "float64": "f8", "complex64": "c8", "complex128": "c16",
}


def _tp_dtype(name: str):
    dtype = getattr(tp, name, None)
    if dtype is None:
        raise NotImplementedError(
            f"TensorPlay does not expose the {name!r} dtype required by this checkpoint"
        )
    return dtype


def _dtype_name_of(tensor) -> str:
    for name in _NUMPY_DTYPES:
        if tensor.dtype == _tp_dtype(name):
            return name
    if tensor.dtype == _tp_dtype("bfloat16"):
        return "bfloat16"
    raise NotImplementedError(f"unsupported TensorPlay dtype: {tensor.dtype}")


def _contiguous_stride(shape: list) -> list:
    stride = [0] * len(shape)
    running = 1
    for index in range(len(shape) - 1, -1, -1):
        stride[index] = running
        running *= shape[index]
    return stride


def _tensor_bytes(tensor) -> bytes:
    """Return the raw contiguous CPU bytes of ``tensor``."""

    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    if tensor.device.type != tp.DeviceType.CPU:
        tensor = tensor.to(tp.Device(tp.DeviceType.CPU))
    dtype_name = _dtype_name_of(tensor)
    if dtype_name == "bfloat16":
        return tensor.view(_tp_dtype("uint16")).numpy().tobytes()
    return tensor.numpy().tobytes()


def _tensor_from_flat_bytes(data: bytes, dtype_name: str):
    numel = len(data) // _ITEMSIZE[dtype_name]
    if numel == 0:
        return None
    if dtype_name == "bfloat16":
        flat = tp.Tensor._from_bytes(data, [numel], _tp_dtype("uint16"))
        return flat.view(_tp_dtype("bfloat16"))
    return tp.Tensor._from_bytes(data, [numel], _tp_dtype(dtype_name))


def _reshape_or_view(flat, size: list, stride: list, storage_offset: int):
    numel = 1
    for dim in size:
        numel *= dim
    if numel == 0 and any(dim == 0 for dim in size):
        return tp.empty(size, dtype=flat.dtype)
    if storage_offset == 0 and stride == _contiguous_stride(size):
        return flat.reshape(size) if size else flat.reshape([])
    return flat.as_strided(size, stride, storage_offset)


def _byteswap_bytes(data: bytes, dtype_name: str) -> bytes:
    import numpy as np

    if dtype_name == "bfloat16":
        dtype_name = "uint16"
    itemsize = np.dtype(_NUMPY_DTYPES[dtype_name]).itemsize
    if itemsize == 1 or not data:
        return data
    array = np.frombuffer(data, dtype=_NUMPY_DTYPES[dtype_name])
    return array.byteswap().tobytes()


def _parse_device(spec: str):
    if spec == "cpu":
        return tp.Device(tp.DeviceType.CPU)
    if ":" in spec:
        kind, index = spec.split(":", 1)
        return tp.Device(getattr(tp.DeviceType, kind.upper()), int(index))
    return tp.Device(getattr(tp.DeviceType, spec.upper()))


def _move_to(tensor, target):
    if target is None or target == "cpu":
        return tensor
    return tensor.to(_parse_device(target))


def _device_string_of(tensor) -> str:
    raw = tensor.device
    if raw.type == tp.DeviceType.CPU:
        return "cpu"
    if raw.type == tp.DeviceType.CUDA:
        return f"cuda:{raw.index or 0}"
    return str(raw)


# ---------------------------------------------------------------------------
# allowlisted unpickler
# ---------------------------------------------------------------------------


class _StorageType:

    def __init__(self, storage_name: str):
        try:
            self.dtype_name = _TORCH_STORAGE_DTYPES[storage_name]
        except KeyError as error:
            raise NotImplementedError(
            ) from error

    def __repr__(self):
        return f"_StorageType({self.dtype_name})"


def _rebuild_tensor(storage, storage_offset, size, stride):
    return _rebuild_tensor_v2(storage, storage_offset, size, stride, False, {})


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad,
                       backward_hooks, metadata=None):
    flat = storage.materialize()
    size = [int(dim) for dim in size]
    stride = [int(dim) for dim in stride]
    return _reshape_or_view(flat, size, stride, int(storage_offset))


_rebuild_tensor_v2.__module__ = "torch._utils"
_rebuild_tensor_v2.__name__ = "_rebuild_tensor_v2"
_rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"
_rebuild_tensor_v2._tp_torch_ref = ("torch._utils", "_rebuild_tensor_v2")
_rebuild_tensor.__module__ = "torch._utils"
_rebuild_tensor.__name__ = "_rebuild_tensor"
_rebuild_tensor.__qualname__ = "_rebuild_tensor"
_rebuild_tensor._tp_torch_ref = ("torch._utils", "_rebuild_tensor")


def _rebuild_parameter(data, requires_grad, backward_hooks, process_dict=None):
    return data


class _WeightsOnlyUnpickler(pickle.Unpickler):
    """Allowlist-based unpickler: never imports or executes unknown code."""

    def __init__(self, file, *, persistent_load, resolve_global):
        super().__init__(file)
        self._persistent_load_fn = persistent_load
        self._resolve_global = resolve_global

    def persistent_load(self, saved_id):
        return self._persistent_load_fn(saved_id)

    def find_class(self, module, name):
        resolved = self._resolve_global(module, name)
        if resolved is not None:
            return resolved
        raise pickle.UnpicklingError(
            f"Unsupported global: GLOBAL {module}.{name} was not allowlisted by "
            "the TensorPlay weights-only loader."
        )


def _make_global_resolver():
    import numpy as np

    def resolve(module: str, name: str):
        if module == "torch":
            if name.endswith("Storage"):
                return _StorageType(name)
            if name in {"strided", "sparse_coo", "sparse_csr", "sparse_csc",
                        "sparse_bsr", "sparse_bsc"}:
                return name
            if name == "_get_layout":
                return lambda layout: layout
            if name in _NUMPY_DTYPES or name == "bfloat16":
                return _tp_dtype(name)
            if name in {"Tensor"}:
                return tp.Tensor
            return None
        if module == "torch._utils":
            if name == "_rebuild_tensor_v2":
                return _rebuild_tensor_v2
            if name == "_rebuild_tensor":
                return _rebuild_tensor
            if name == "_rebuild_parameter":
                return _rebuild_parameter
            if name == "_rebuild_sparse_tensor":
                raise NotImplementedError(
                )
            return None
        if module == "torch.nn.parameter" and name == "_rebuild_parameter":
            return _rebuild_parameter
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "numpy" and name == "dtype":
            return np.dtype
        if module in {"numpy.core.multiarray", "numpy._core.multiarray"} and name == "_reconstruct":
            return np.core.multiarray._reconstruct
        return None

    return resolve


def _convert_numpy_leaves(obj):
    """Replace numpy arrays left in an unpickled structure by tp tensors."""

    import numpy as np

    if isinstance(obj, np.ndarray):
        dtype_name = str(obj.dtype.name)
        if dtype_name not in _NUMPY_DTYPES:
            raise NotImplementedError(f"numpy dtype {dtype_name} is not supported")
        data = np.ascontiguousarray(obj).tobytes()
        flat = _tensor_from_flat_bytes(data, dtype_name)
        return _reshape_or_view(flat, list(obj.shape), _contiguous_stride(list(obj.shape)), 0)
    if isinstance(obj, dict):
        return {key: _convert_numpy_leaves(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_convert_numpy_leaves(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_convert_numpy_leaves(value) for value in obj)
    return obj


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


class _LazyZipStorage:
    def __init__(self, archive: zipfile.ZipFile, reader: "_ReaderState",
                 key: str, dtype_name: str, location: str, numel: int):
        self._archive = archive
        self._reader = reader
        self.key = key
        self.dtype_name = dtype_name
        self.location = location
        self.numel = max(numel, 0)
        self._flat = None

    def materialize(self):
        if self._flat is None:
            record = f"data/{self.key}"
            data = self._archive.read(record)
            expected = self.numel * _ITEMSIZE[self.dtype_name]
            if len(data) != expected:
                raise ValueError(
                    f"bytes, expected {expected}"
                )
            if self._reader.swap_bytes:
                data = _byteswap_bytes(data, self.dtype_name)
            flat = _tensor_from_flat_bytes(data, self.dtype_name)
            if flat is None:
                flat = tp.empty((0,), dtype=_tp_dtype(self.dtype_name))
            self._flat = _move_to(flat, self._reader.resolve_location(self.location))
        return self._flat


class _ReaderState:
    def __init__(self, map_location):
        self.map_location = map_location
        self.swap_bytes = False

    def resolve_location(self, location: str):
        from .serialization import resolve_map_location

        return resolve_map_location(self.map_location, location)


class _RootedZipReader:

    def __init__(self, archive: zipfile.ZipFile, prefix: str):
        self._archive = archive
        self._prefix = prefix

    def namelist(self) -> list:
        cut = len(self._prefix)
        return [name[cut:] if name.startswith(self._prefix) else name
                for name in self._archive.namelist()]

    def read(self, name: str) -> bytes:
        return self._archive.read(self._prefix + name)


def _normalize_torch_zip_root(archive: zipfile.ZipFile) -> zipfile.ZipFile:
    names = set(archive.namelist())
    if "data.pkl" in names:
        return archive
    roots = {name[:-len("data.pkl")] for name in names if name.endswith("/data.pkl")}
    if len(roots) == 1:
        return _RootedZipReader(archive, next(iter(roots)))
    return archive


def _read_zip_archive(archive: zipfile.ZipFile, *, map_location) -> Any:
    archive = _normalize_torch_zip_root(archive)
    names = set(archive.namelist())
    if "constants.pkl" in names:
        raise RuntimeError(
            "modules. Export the weights as a state_dict instead."
        )
    if "data.pkl" not in names:
        raise ValueError("not a legacy checkpoint: missing data.pkl record")

    state = _ReaderState(map_location)
    if "byteorder" in names:
        byteorder = archive.read("byteorder").decode("ascii")
        if byteorder not in {"little", "big"}:
            raise ValueError(f"unknown byteorder record: {byteorder!r}")
        state.swap_bytes = byteorder != sys.byteorder

    def persistent_load(saved_id):
        if not isinstance(saved_id, tuple) or not saved_id or saved_id[0] != "storage":
            raise pickle.UnpicklingError(f"unsupported persistent id: {saved_id!r}")
        storage_type, key, location, numel = saved_id[1:]
        if isinstance(location, bytes):
            location = location.decode("ascii")
        return _LazyZipStorage(
            archive, state, str(key), storage_type.dtype_name, str(location), int(numel)
        )

    unpickler = _WeightsOnlyUnpickler(
        io.BytesIO(archive.read("data.pkl")),
        persistent_load=persistent_load,
        resolve_global=_make_global_resolver(),
    )
    return _convert_numpy_leaves(unpickler.load())


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def _read_magic_number_stream(fileobj: BinaryIO, *, map_location) -> Any:
    state = _ReaderState(map_location)

    trivial_resolver = lambda module, name: None
    magic = _WeightsOnlyUnpickler(
        fileobj, persistent_load=lambda saved_id: None,
        resolve_global=trivial_resolver,
    ).load()
    if magic != TORCH_MAGIC_NUMBER:
        raise ValueError(f"bad magic number {magic!r}; corrupt legacy checkpoint?")
    protocol_version = _WeightsOnlyUnpickler(
        fileobj, persistent_load=lambda saved_id: None,
        resolve_global=trivial_resolver,
    ).load()
    if protocol_version != TORCH_PROTOCOL_VERSION:
        raise ValueError(f"unsupported legacy protocol version {protocol_version!r}")
    sys_info = _WeightsOnlyUnpickler(
        fileobj, persistent_load=lambda saved_id: None,
        resolve_global=trivial_resolver,
    ).load()
    if isinstance(sys_info, dict):
        little_endian = bool(sys_info.get("little_endian", True))
        state.swap_bytes = little_endian != (sys.byteorder == "little")

    pending: "OrderedDict[str, dict]" = OrderedDict()
    deferred_views: list = []

    def persistent_load(saved_id):
        if not isinstance(saved_id, tuple) or not saved_id or saved_id[0] != "storage":
            if saved_id and saved_id[0] == "module":
                return saved_id[1]
            raise pickle.UnpicklingError(f"unsupported persistent id: {saved_id!r}")
        data = saved_id[1:]
        storage_type, key, location, numel = data[0], str(data[1]), data[2], data[3]
        view_metadata = data[4] if len(data) > 4 else None
        if isinstance(location, bytes):
            location = location.decode("ascii")
        slot = pending.get(key)
        if slot is None:
            # Payload bytes trail the pickle stream, so allocate the backing
            # fill it in place afterwards -- rebuilt views reference it.
            dtype_name = storage_type.dtype_name
            target = state.resolve_location(str(location))
            numel = max(int(numel), 0)
            if target is not None and target != "cpu":
                tensor = tp.empty((numel,), dtype=_tp_dtype(dtype_name),
                                  device=_parse_device(target))
            else:
                tensor = tp.empty((numel,), dtype=_tp_dtype(dtype_name))
            slot = {"dtype": dtype_name, "location": str(location),
                    "numel": numel, "tensor": tensor}
            pending[key] = slot
        if view_metadata is not None:
            view_key, offset, view_numel = view_metadata
            deferred_views.append((str(view_key), key, int(offset), int(view_numel)))
        return _PendingStorage(slot, state)

    unpickler = _WeightsOnlyUnpickler(
        fileobj,
        persistent_load=persistent_load,
        resolve_global=_make_global_resolver(),
    )
    result = unpickler.load()

    # Storage payloads follow the pickle stream, consumed in the order of the
    stored_keys = _WeightsOnlyUnpickler(
        fileobj, persistent_load=lambda saved_id: None,
        resolve_global=trivial_resolver,
    ).load()

    def fill(slot):
        (stored_numel,) = struct.unpack("<q", fileobj.read(8))
        itemsize = _ITEMSIZE[slot["dtype"]]
        nbytes = stored_numel * itemsize
        data = fileobj.read(nbytes)
        if len(data) != nbytes:
            raise ValueError("truncated legacy checkpoint storage")
        if state.swap_bytes:
            data = _byteswap_bytes(data, slot["dtype"])
        incoming = _tensor_from_flat_bytes(data, slot["dtype"])
        if incoming is None:
            return
        slot["tensor"].copy_(incoming.reshape(slot["tensor"].shape))

    for key in stored_keys:
        key = str(key)
        if key in pending:
            fill(pending[key])
    for view_key, root_key, offset, view_numel in deferred_views:
        base = pending[root_key]["tensor"]
        pending[view_key] = {
            "dtype": pending[root_key]["dtype"],
            "location": pending[root_key]["location"],
            "numel": view_numel,
            "tensor": base.as_strided([view_numel], [1], offset),
        }
    return _convert_numpy_leaves(result)


class _PendingStorage:
    def __init__(self, slot: dict, state: _ReaderState):
        self.slot = slot
        self.state = state

    def materialize(self):
        return self.slot["tensor"]


# ---------------------------------------------------------------------------
# ancient tar format (pre-0.1, reading only)
# ---------------------------------------------------------------------------


def _read_tar_archive(archive: tarfile.TarFile, *, map_location) -> Any:
    state = _ReaderState(map_location)
    slots: OrderedDict[str, dict] = OrderedDict()

    with archive.extractfile("storages") as handle:
        stream = io.BytesIO(handle.read())
        num_storages = pickle.load(stream)
        for _ in range(num_storages):
            key, location, storage_type = pickle.load(stream)
            (numel,) = struct.unpack("<q", stream.read(8))
            dtype_name = storage_type.dtype_name
            nbytes = int(numel) * _ITEMSIZE[dtype_name]
            data = stream.read(nbytes)
            if len(data) != nbytes:
                raise ValueError(f"truncated legacy tar storage {key!r}")
            if state.swap_bytes:
                data = _byteswap_bytes(data, dtype_name)
            flat = _tensor_from_flat_bytes(data, dtype_name)
            if flat is None:
                flat = tp.empty((max(int(numel), 0),), dtype=_tp_dtype(dtype_name))
            slots[str(key)] = {
                "tensor": _move_to(flat, state.resolve_location(str(location))),
            }

    specs = []
    with archive.extractfile("tensors") as handle:
        stream = io.BytesIO(handle.read())
        num_tensors = pickle.load(stream)
        for _ in range(num_tensors):
            key, storage_id, _original_type = pickle.load(stream)
            (ndim,) = struct.unpack("<i", stream.read(4))
            stream.read(4)  # legacy treated ndim as 8 bytes
            size = list(struct.unpack(f"<{ndim}q", stream.read(8 * ndim))) if ndim else []
            stride = list(struct.unpack(f"<{ndim}q", stream.read(8 * ndim))) if ndim else []
            (storage_offset,) = struct.unpack("<q", stream.read(8))
            specs.append((str(key), str(storage_id), size, stride, int(storage_offset)))

    rebuilt = {}
    for key, storage_id, size, stride, storage_offset in specs:
        flat = slots[storage_id]["tensor"]
        rebuilt[key] = _reshape_or_view(flat, size, stride, storage_offset)

    def persistent_load(saved_id):
        if isinstance(saved_id, tuple):
            if saved_id and saved_id[0] == "module":
                return saved_id[1]
            raise pickle.UnpicklingError(f"unsupported persistent id: {saved_id!r}")
        return rebuilt[str(saved_id)]

    with archive.extractfile("pickle") as handle:
        unpickler = _WeightsOnlyUnpickler(
            handle,
            persistent_load=persistent_load,
            resolve_global=_make_global_resolver(),
        )
        result = unpickler.load()
    return _convert_numpy_leaves(result)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def read_torch_file(fileobj: BinaryIO, *, map_location=None) -> Any:

    head = fileobj.read(4)
    fileobj.seek(0)
    if head == b"PK\x03\x04":
        with zipfile.ZipFile(fileobj) as archive:
            return _read_zip_archive(archive, map_location=map_location)

    try:
        with tarfile.open(fileobj=fileobj, mode="r:") as archive:
            member_names = {member.name for member in archive.getmembers()}
            if {"storages", "tensors", "pickle"} <= member_names:
                return _read_tar_archive(archive, map_location=map_location)
    except (tarfile.TarError, EOFError, KeyError):
        pass

    fileobj.seek(0)
    try:
        probe = pickle.Unpickler(fileobj).load()
    except Exception:
        probe = None
    finally:
        fileobj.seek(0)
    if probe == TORCH_MAGIC_NUMBER:
        return _read_magic_number_stream(fileobj, map_location=map_location)

    raise ValueError(
    )


class _PTStorageRef:

    __slots__ = ("key",)

    def __init__(self, key: str):
        self.key = key


def _make_pt_storage_class(dtype_name: str):
    storage_name = _STORAGE_NAMES_BY_DTYPE[dtype_name]

    cls = type(
        storage_name,
        (),
        {"__module__": "torch", "__qualname__": storage_name, "__name__": storage_name},
    )
    cls._tp_torch_ref = ("torch", storage_name)
    return cls


_PT_STORAGE_CLASSES = {
    dtype_name: _make_pt_storage_class(dtype_name) for dtype_name in _STORAGE_NAMES_BY_DTYPE
}


class _TorchCompatPickler(pickle._Pickler):
    """

    CPython's C pickler verifies forged ``__module__``/``__qualname__``
    cross-loading.  Objects advertise their target through a
    """

    def save_global(self, obj, name=None):
        forced = getattr(obj, "_tp_torch_ref", None)
        if forced is not None:
            module, forced_name = forced
            self.write(pickle.GLOBAL
                       + module.encode("ascii") + b"\n"
                       + forced_name.encode("ascii") + b"\n")
            self.memoize(obj)
            return
        super().save_global(obj, name)

    dispatch = dict(pickle._Pickler.dispatch)
    dispatch[FunctionType] = save_global


def write_torch_file(fileobj: BinaryIO, obj: Any, *, pickle_protocol: int = 2) -> None:
    """

    Any picklable object graph is accepted.  TensorPlay tensors are emitted
    """

    storages: dict[int, dict] = {}
    order: list[dict] = []

    def storage_record(tensor) -> _PTStorageRef:
        contiguous = tensor.is_contiguous()
        nbytes = int(tensor.numel()) * int(tensor.itemsize())
        identity = (int(tensor.data_ptr()), nbytes) if contiguous and nbytes > 0 else None
        if identity is not None and identity in storages:
            return _PTStorageRef(storages[identity]["key"])
        record = {
            "key": str(len(order)),
            "tensor": tensor,
            "dtype": _dtype_name_of(tensor),
            "location": _device_string_of(tensor),
            "numel": int(tensor.numel()),
        }
        order.append(record)
        if identity is not None:
            storages[identity] = record
        return _PTStorageRef(record["key"])

    def reduce_tensor(tensor):
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        ref = storage_record(tensor)
        shape = [int(dim) for dim in tensor.shape]
        return (
            _rebuild_tensor_v2,
            (ref, 0, shape, _contiguous_stride(shape), False, {}),
        )

    def persistent_id(obj):
        if isinstance(obj, _PTStorageRef):
            record = next(item for item in order if item["key"] == obj.key)
            storage_cls = _PT_STORAGE_CLASSES[record["dtype"]]
            return ("storage", storage_cls, record["key"], record["location"], record["numel"])
        return None

    data_buf = io.BytesIO()
    pickler = _TorchCompatPickler(data_buf, pickle_protocol)
    dispatch = copyreg.dispatch_table.copy()
    dispatch[tp.Tensor] = reduce_tensor
    parameter_cls = getattr(tp.nn, "Parameter", None)
    if parameter_cls is not None and parameter_cls is not tp.Tensor:
        dispatch[parameter_cls] = reduce_tensor
    pickler.dispatch_table = dispatch
    pickler.persistent_id = persistent_id
    pickler.dump(obj)

    with zipfile.ZipFile(fileobj, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        data_value = data_buf.getvalue()
        archive.writestr("archive/data.pkl", data_value)
        archive.writestr("archive/version", "3")
        archive.writestr("archive/.format_version", "1")
        archive.writestr("archive/.storage_alignment", "64")
        archive.writestr("archive/byteorder", sys.byteorder)
        for record in order:
            archive.writestr(f"archive/data/{record['key']}", _tensor_bytes(record["tensor"]))
        fileobj.flush()


def describe_torch_file(fileobj: BinaryIO) -> dict:

    head = fileobj.read(4)
    fileobj.seek(0)
    if head != b"PK\x03\x04":
        return {"format": "torch_stream_or_tar"}

    with zipfile.ZipFile(fileobj) as archive:
        archive = _normalize_torch_zip_root(archive)
        names = set(archive.namelist())
        if "constants.pkl" in names:
            return {"format": "torchscript_zip"}
        if "data.pkl" not in names:
            return {"format": "unknown_zip"}
        storages: dict[str, dict] = {}

        def persistent_load(saved_id):
            _, storage_type, key, location, numel = saved_id
            storages[str(key)] = {
                "dtype": storage_type.dtype_name,
                "numel": int(numel),
                "location": location.decode() if isinstance(location, bytes) else str(location),
            }
            return None

        def rebuild_stub(storage, storage_offset, size, stride, *args):
            return {"shape": [int(dim) for dim in size],
                    "dtype": getattr(storage, "dtype_name", None)}

        resolver = _make_inspect_resolver(rebuild_stub)
        unpickler = _WeightsOnlyUnpickler(
            io.BytesIO(archive.read("data.pkl")),
            persistent_load=persistent_load,
            resolve_global=resolver,
        )
        structure = unpickler.load()
        info = {
            "format": "torch_zip",
            "byteorder": archive.read("byteorder").decode("ascii")
            if "byteorder" in names else None,
            "storages": storages,
            "structure": _summarize(structure),
        }
        return info


def _make_inspect_resolver(rebuild_stub):
    def resolve(module: str, name: str):
        if module == "torch" and name.endswith("Storage"):
            return _StorageType(name)
        if module == "torch._utils" and name in {"_rebuild_tensor_v2", "_rebuild_tensor"}:
            return rebuild_stub
        if module == "torch.nn.parameter" and name == "_rebuild_parameter":
            return lambda data, *args, **kwargs: data
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "numpy" and name == "dtype":
            import numpy as np

            return np.dtype
        if module in {"numpy.core.multiarray", "numpy._core.multiarray"} and name == "_reconstruct":
            import numpy as np

            return lambda *args, **kwargs: None
        return None

    return resolve


def _summarize(obj, depth: int = 0):
    if depth > 6:
        return "..."
    if isinstance(obj, _StorageType):
        return repr(obj)
    if isinstance(obj, dict):
        return {str(key): _summarize(value, depth + 1) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_summarize(value, depth + 1) for value in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return type(obj).__name__


# ---------------------------------------------------------------------------
# safetensors
# ---------------------------------------------------------------------------

_SAFETENSORS_DTYPES = {
    "BOOL": "bool", "U8": "uint8", "I8": "int8", "I16": "int16", "I32": "int32",
    "I64": "int64", "U32": "uint32", "U64": "uint64", "F16": "float16",
    "BF16": "bfloat16", "F32": "float32", "F64": "float64",
}

_SAFETENSORS_DTYPE_NAMES = {v: k for k, v in _SAFETENSORS_DTYPES.items()}


def _read_safetensors_header(fileobj: BinaryIO) -> tuple[dict, int]:
    raw_length = fileobj.read(8)
    if len(raw_length) != 8:
        raise ValueError("truncated safetensors file: missing header length")
    header_length = struct.unpack("<Q", raw_length)[0]
    header_bytes = fileobj.read(header_length)
    if len(header_bytes) != header_length:
        raise ValueError("truncated safetensors file: truncated header")
    try:
        header = json.loads(header_bytes)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid safetensors header JSON: {error}") from error
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be a JSON object")
    return header, 8 + header_length


def read_safetensors_file(fileobj: BinaryIO, *, map_location=None, mmap: bool = False) -> "OrderedDict[str, Any]":
    """Load a safetensors file as an ``OrderedDict`` of TensorPlay tensors.

    With ``mmap=True`` tensors become zero-copy views over a private mapping
    are paged in on first touch and never duplicated into anonymous memory.
    The returned dict keeps the mapping alive.  Falls back to an eager read
    when the file object has no real file descriptor or the platform needs a
    byte swap.  File-level ``__metadata__`` is surfaced by
    :func:`tensorplay.serialization.inspect_checkpoint`, not injected into the
    """

    header, data_start = _read_safetensors_header(fileobj)
    metadata = header.pop("__metadata__", {})

    target = None
    if map_location is not None:
        from .serialization import resolve_map_location

        target = resolve_map_location(map_location, "cpu")

    mapping = None
    if mmap and sys.byteorder == "little":
        import mmap as _mmap

        try:
            fileno = fileobj.fileno()
        except (AttributeError, OSError, io.UnsupportedOperation):
            fileno = None
        if fileno is not None and fileno >= 0:
            mapping = _mmap.mmap(fileno, 0, access=_mmap.ACCESS_COPY)

    import numpy as np

    result: "OrderedDict[str, Any]" = OrderedDict()
    cursor = 0
    for name, info in header.items():
        start, end = info["data_offsets"]
        if not (cursor <= start <= end):
            raise ValueError(
                f"safetensors tensor {name!r} has non-monotonic data_offsets "
                f"[{start}, {end})"
            )
        cursor = end
        dtype_name = _SAFETENSORS_DTYPES.get(info["dtype"])
        if dtype_name is None:
            raise NotImplementedError(
                f"unsupported safetensors dtype: {info['dtype']!r}"
            )
        shape = [int(dim) for dim in info["shape"]]
        numel = 1
        for dim in shape:
            numel *= dim

        tensor = None
        if mapping is not None and numel > 0:
            tensor = _safetensors_mmap_tensor(mapping, data_start + start, end - start,
                                              shape, dtype_name)
        if tensor is None:
            data = _read_exact(fileobj, data_start + start, end - start)
            if sys.byteorder == "big":
                data = _byteswap_bytes(data, dtype_name)
            flat = _tensor_from_flat_bytes(data, dtype_name)
            tensor = _reshape_or_view(flat, shape, _contiguous_stride(shape), 0)
        result[name] = _move_to(tensor, target)

    if mapping is not None:
        holder = _KeepAliveDict(result)
        holder._keepalive = mapping
        return holder
    return result


def _read_exact(fileobj: BinaryIO, offset: int, length: int) -> bytes:
    fileobj.seek(offset)
    data = fileobj.read(length)
    if len(data) != length:
        raise ValueError("truncated safetensors payload")
    return data


def _safetensors_mmap_tensor(mapping, offset: int, length: int, shape: list,
                             dtype_name: str):
    """Zero-copy tensor over ``mapping[offset:offset+length]`` via DLPack.

    Chain: mmap -> numpy view (no copy) -> ``tp.from_dlpack`` (no copy).  The
    DLPack capsule keeps the numpy array (and through it the mapping) alive.
    Returns None when the zero-copy path is unavailable so callers can fall
    back to an eager read.
    """

    if dtype_name == "bfloat16":
        np_dtype = "u2"
    elif dtype_name in _NUMPY_DTYPES:
        np_dtype = _NUMPY_DTYPES[dtype_name]
    else:
        return None
    import numpy as np

    window = memoryview(mapping)[offset:offset + length]
    array = np.frombuffer(window, dtype=np_dtype)
    try:
        try:
            tensor = tp.from_dlpack(array)
        except (TypeError, AttributeError):
            tensor = tp.from_dlpack(array.__dlpack__())
    except Exception:
        return None
    if dtype_name == "bfloat16":
        tensor = tensor.view(_tp_dtype("bfloat16"))
    return _reshape_or_view(tensor, shape, _contiguous_stride(shape), 0)


class _KeepAliveDict(OrderedDict):
    """OrderedDict pinning a backing mapping for zero-copy tensor views."""

    _keepalive = None


def write_safetensors_file(fileobj: BinaryIO, obj: Mapping[str, Any], *,
                           metadata: Mapping[str, str] | None = None) -> None:
    """Write a flat mapping of name -> tensor as a safetensors file."""

    if not isinstance(obj, Mapping):
        raise TypeError(
            "safetensors stores a flat name->tensor mapping only; use '.mega' "
            "or '.pt' for nested containers"
        )

    header: dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = {str(key): str(value) for key, value in metadata.items()}
    prepared = []
    offset = 0
    for name, tensor in obj.items():
        if not isinstance(tensor, tp.Tensor):
            raise TypeError(
                f"safetensors requires a flat mapping of name to Tensor; "
                f"got non-tensor value at {name!r}"
            )
        dtype_name = _dtype_name_of(tensor)
        st_name = _SAFETENSORS_DTYPE_NAMES.get(dtype_name)
        if st_name is None:
            raise NotImplementedError(
                f"dtype {dtype_name!r} is not representable in safetensors"
            )
        nbytes = int(tensor.numel()) * int(tensor.itemsize())
        header[str(name)] = {
            "dtype": st_name,
            "shape": [int(dim) for dim in tensor.shape],
            "data_offsets": [offset, offset + nbytes],
        }
        prepared.append((tensor, nbytes))
        offset += nbytes

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padding = (8 - len(header_bytes) % 8) % 8
    header_bytes += b" " * padding

    fileobj.write(struct.pack("<Q", len(header_bytes)))
    fileobj.write(header_bytes)
    for tensor, nbytes in prepared:
        data = _tensor_bytes(tensor)
        if len(data) != nbytes:
            raise RuntimeError(
                f"internal error: expected {nbytes} payload bytes, got {len(data)}"
            )
        fileobj.write(data)
    fileobj.flush()


def describe_safetensors_file(fileobj: BinaryIO) -> dict:
    header, _data_start = _read_safetensors_header(fileobj)
    metadata = header.pop("__metadata__", {})
    tensors = {}
    total = 0
    for name, info in header.items():
        start, end = info["data_offsets"]
        total = max(total, end)
        tensors[name] = {
            "shape": [int(dim) for dim in info["shape"]],
            "dtype": info["dtype"],
            "nbytes": end - start,
        }
    return {"format": "safetensors", "metadata": metadata, "tensors": tensors,
            "payload_nbytes": total}
