# mypy: allow-untyped-defs
"""cuDNN backend property helpers.

Exposes ``tensorplay.backends.cudnn.allow_tf32`` (defaults to True, like
which times candidate convolution algorithms per shape and caches the
fastest.
"""

import sys

import tensorplay
from tensorplay.backends import ContextProp, PropModule

__all__ = [
    "allow_tf32",
    "benchmark",
    "is_available",
    "version",
]


def is_available() -> bool:
    r"""Returns a bool indicating if CUDNN is currently available."""
    return tensorplay.backends.cuda.is_available()


def version() -> int:
    r"""Returns the version of cuDNN, or -1 when unavailable."""
    try:
        return int(tensorplay._C._cuda.get_cudnn_version())
    except AttributeError:
        return -1


class _CuDNNModule(PropModule):
    allow_tf32 = ContextProp(
        tensorplay._C._get_cudnn_allow_tf32,
        tensorplay._C._set_cudnn_allow_tf32,
    )
    benchmark = ContextProp(
        tensorplay._C._get_cudnn_benchmark,
        tensorplay._C._set_cudnn_benchmark,
    )


sys.modules[__name__] = _CuDNNModule(sys.modules[__name__], __name__)
