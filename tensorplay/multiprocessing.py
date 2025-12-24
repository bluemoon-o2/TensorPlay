import multiprocessing
import tensorplay
from multiprocessing import *
from multiprocessing.reduction import ForkingPickler

# This module wraps python's multiprocessing to provide support for
# shared memory passing of Tensor objects. 

def reduce_tensor(t):
    if not t.is_shared():
        t.share_memory_()
    # Manually construct reduce tuple to bypass broken __reduce__ in nanobind bindings
    # (reconstructor, args, state)
    # Tensor() constructor takes args, but we use __setstate__ to initialize
    # So we pass empty args to constructor (which creates empty tensor) and then state
    return t.__class__, (), t.__getstate__()

# Register for TensorBase too since Tensor is an alias
try:
    from tensorplay._C import TensorBase
    ForkingPickler.register(TensorBase, reduce_tensor)
except ImportError:
    pass

ForkingPickler.register(tensorplay.Tensor, reduce_tensor)

__all__ = ['get_context', 'Queue', 'Event', 'Process', 'current_process', 'active_children']

def get_context(method=None):
    return multiprocessing.get_context(method)
