"""Kernel codegen cache (L5-M1).

Modeled on ``torch/_inductor/codecache.py``: content-addressed on-disk cache
for compiled artifacts. The lowering backend (stax/Triton/AVX) supplies the
generated ``source`` plus its compile callable; this module owns key
canonicalization, lookup, atomic publication and process-level memoization,
so backends stay free of cache plumbing.

Layout: ``$TP_CACHE_DIR`` or ``<cwd>/.tp_cache/kernels/<backend>/ab/<key>.<ext>``
where ``ab`` is the first two hex digits of the key (fan-out like torch's
local cache). Publication is temp-file + rename, so concurrent processes
never observe partial artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from typing import Any, Callable, Dict, Optional, Tuple


class CodeCache:
    """Content-addressed store mapping (source, entry, options) -> artifact."""

    def __init__(self, backend: str, root: Optional[str] = None) -> None:
        self.backend = backend
        env_root = os.environ.get("TP_CACHE_DIR")
        base = root or env_root or os.path.join(os.getcwd(), ".tp_cache")
        self.root = os.path.join(base, "kernels", backend)

    # -- keying -------------------------------------------------------------

    @staticmethod
    def _canonical(options: Optional[Dict[str, Any]]) -> str:
        return json.dumps(options or {}, sort_keys=True, default=repr)

    def cache_key(
        self,
        source: str,
        entry: str = "",
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        h = hashlib.sha256()
        h.update(self.backend.encode())
        h.update(b"\x00")
        h.update(entry.encode())
        h.update(b"\x00")
        h.update(self._canonical(options).encode())
        h.update(b"\x00")
        h.update(source.encode())
        return h.hexdigest()

    def path_for(self, key: str, ext: str = "bin") -> str:
        folder = os.path.join(self.root, key[:2])
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, f"{key}.{ext}")

    # -- storage ------------------------------------------------------------

    def load(self, key: str, ext: str = "bin") -> Optional[bytes]:
        memo = self._memo().get((key, ext))
        if memo is not None:
            return memo
        path = self.path_for(key, ext)
        try:
            with open(path, "rb") as fh:
                data = fh.read()
        except OSError:
            return None
        self._memo()[(key, ext)] = data
        return data

    def store(self, key: str, payload: bytes, ext: str = "bin") -> str:
        path = self.path_for(key, ext)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(payload)
            os.replace(tmp, path)  # atomic publication
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        self._memo()[(key, ext)] = payload
        return path

    def compile_or_load(
        self,
        compile_fn: Callable[[str], bytes],
        source: str,
        *,
        entry: str = "",
        options: Optional[Dict[str, Any]] = None,
        ext: str = "bin",
    ) -> Tuple[bytes, str]:
        """Return ``(artifact, path)``, compiling through ``compile_fn`` on miss."""

        key = self.cache_key(source, entry, options)
        cached = self.load(key, ext)
        if cached is not None:
            return cached, self.path_for(key, ext)
        artifact = compile_fn(source)
        self.store(key, artifact, ext)
        return artifact, self.path_for(key, ext)

    # -- process-level memo ---------------------------------------------------

    def _memo(self) -> Dict[Tuple[str, str], bytes]:
        memo = getattr(self, "_memo_dict", None)
        if memo is None:
            memo = {}
            self._memo_dict = memo
        return memo


_default_caches: Dict[str, CodeCache] = {}


def default_cache(backend: str) -> CodeCache:
    """Process-wide cache instance per backend."""

    cache = _default_caches.get(backend)
    if cache is None:
        cache = CodeCache(backend)
        _default_caches[backend] = cache
    return cache
