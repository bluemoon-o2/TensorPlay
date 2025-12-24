from typing import Optional, Union, Any

# Core functions from C++ extension
from ._C._cuda import (
    is_available,
    device_count,
    empty_cache,
    current_device as _current_device,
    set_device as _set_device,
    get_device_name as _get_device_name,
    get_device_capability as _get_device_capability,
    get_device_properties as _get_device_properties,
    memory_allocated as _memory_allocated,
    max_memory_allocated as _max_memory_allocated,
    reset_max_memory_allocated as _reset_max_memory_allocated,
    synchronize as _synchronize,
    manual_seed as _manual_seed
)

_initialized = False

def init():
    """Initialize CUDA state. This is lazy initialized so normally not needed."""
    global _initialized
    if not is_available():
        return
    # Force initialization
    _synchronize(-1)
    _initialized = True

def is_initialized():
    """Check if CUDA has been initialized."""
    return _initialized

def current_device() -> int:
    """Return the index of a currently selected device."""
    return _current_device()

def set_device(device: Union[int, Any]):
    """Sets the current device."""
    if hasattr(device, 'index'):
        device = device.index
    _set_device(device)

def get_device_name(device: Optional[Union[int, Any]] = None) -> str:
    """Gets the name of a device."""
    if device is None:
        device = current_device()
    elif hasattr(device, 'index'):
        device = device.index
    return _get_device_name(device)

def get_device_capability(device: Optional[Union[int, Any]] = None) -> tuple:
    """Gets the cuda capability of a device."""
    if device is None:
        device = current_device()
    elif hasattr(device, 'index'):
        device = device.index
    return _get_device_capability(device)

def get_device_properties(device: Union[int, Any]):
    """Gets the properties of a device."""
    if hasattr(device, 'index'):
        device = device.index
    return _get_device_properties(device)

def synchronize(device: Optional[Union[int, Any]] = None):
    """Waits for all kernels in all streams on a CUDA device to complete."""
    if device is None:
        device = -1
    elif hasattr(device, 'index'):
        device = device.index
    _synchronize(device)

# Memory Management
def memory_allocated(device: Optional[Union[int, Any]] = None) -> int:
    """Returns the current GPU memory usage by tensors in bytes for a given device."""
    if device is None:
        device = current_device()
    elif hasattr(device, 'index'):
        device = device.index
    return _memory_allocated(device)

def max_memory_allocated(device: Optional[Union[int, Any]] = None) -> int:
    """Returns the maximum GPU memory usage by tensors in bytes for a given device."""
    if device is None:
        device = current_device()
    elif hasattr(device, 'index'):
        device = device.index
    return _max_memory_allocated(device)

def reset_max_memory_allocated(device: Optional[Union[int, Any]] = None):
    """Resets the starting point for tracking maximum GPU memory usage."""
    if device is None:
        device = current_device()
    elif hasattr(device, 'index'):
        device = device.index
    _reset_max_memory_allocated(device)

def memory_reserved(device: Optional[Union[int, Any]] = None) -> int:
    """Returns the current GPU memory managed by the caching allocator in bytes for a given device."""
    # In simple allocator, reserved == allocated
    return memory_allocated(device)

def max_memory_reserved(device: Optional[Union[int, Any]] = None) -> int:
    """Returns the maximum GPU memory managed by the caching allocator in bytes for a given device."""
    return max_memory_allocated(device)

def reset_max_memory_reserved(device: Optional[Union[int, Any]] = None):
    """Resets the starting point for tracking maximum GPU memory managed by the caching allocator."""
    reset_max_memory_allocated(device)

# Streams and Events (Mock/Minimal Implementation)
class Event:
    """Wrapper around a CUDA event."""
    def __init__(self, enable_timing=False, blocking=False, interprocess=False):
        self.enable_timing = enable_timing
        self.blocking = blocking
        self.interprocess = interprocess
        
    def record(self, stream=None):
        pass
        
    def wait(self, stream=None):
        pass
        
    def query(self):
        return True
        
    def elapsed_time(self, end_event):
        return 0.0
        
    def synchronize(self):
        pass

class Stream:
    """Wrapper around a CUDA stream."""
    def __init__(self, device=None, priority=0, **kwargs):
        self.device = device
        self.priority = priority
        
    def wait_event(self, event):
        pass
        
    def wait_stream(self, stream):
        pass
        
    def record_event(self, event=None):
        return event if event else Event()
        
    def synchronize(self):
        synchronize(self.device)
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        pass

def manual_seed(seed: int):
    """Sets the seed for generating random numbers for the current GPU."""
    _manual_seed(seed)

def manual_seed_all(seed: int):
    """Sets the seed for generating random numbers on all GPUs."""
    _manual_seed(seed)

def cudart():
    """Returns the ctypes wrapper around the CUDA runtime DLL."""
    return None
