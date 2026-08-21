"""Lazy registration of built-in TensorPlay compiler backends."""

from __future__ import annotations

from .registry import register_backend


def register() -> None:
    from ..backends.stax import stax

    register_backend(stax, name="stax")
