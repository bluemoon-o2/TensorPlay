"""TorchScript compatibility shims.

torchvision model files use ``@torch.jit.unused`` / ``torch.jit.is_scripting()``
(e.g. googlenet, inception) to guard eager paths.  TensorPlay has no
TorchScript, so the decorators are identity passthroughs and
``is_scripting()`` always returns False — exactly the semantics of an eager
execution environment.
"""

from typing import Callable, TypeVar

T = TypeVar("T")


def is_scripting() -> bool:
    """Returns False: TensorPlay always executes eagerly (cf. torch.jit.is_scripting)."""
    return False


def is_tracing() -> bool:
    return False


def is_exporting() -> bool:
    return False


def is_importing() -> bool:
    return False


def unused(fn: T) -> T:
    """Identity decorator; marks a method as unavailable under TorchScript."""
    return fn


def script(fn=None, *args, **kwargs):
    """No-op: returns the function unchanged (eager execution)."""
    if fn is None:
        return lambda f: f
    return fn


def ignore(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return lambda f: f


def _overload_method(*args, **kwargs):
    """Identity decorator standing in for torch.jit._overload_method."""
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return lambda f: f


def export(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return lambda f: f


def interface(fn: T) -> T:
    return fn


def Final(value):
    return value


def isinstance(x, *args):
    """Eager fallback for torch.jit.isinstance: always the else-branch value."""
    if len(args) == 2:
        return False
    return False
