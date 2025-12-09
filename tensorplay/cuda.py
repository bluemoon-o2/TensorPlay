from . import _C

def is_available():
    return _C.cuda_is_available()
