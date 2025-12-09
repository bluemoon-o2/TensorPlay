from . import _C
import tensorplay

# Re-export Tensor class
# In the future, we can extend this class or wrap it to add pure Python functionality
Tensor = _C.TensorBase

# monkey-patching
def is_float(self):
    return self.dtype == _C.DType.float32 or self.dtype == _C.DType.float64

def tolist(self):
    """
    Convert tensor to (nested) list.
    """
    if self.dim() == 0:
        return self.item()
    
    res = []
    # If 1D, we can iterate
    if self.dim() == 1:
        for i in range(self.size(0)):
            res.append(self[i].item())
        return res
    
    # If > 1D, recurse
    for i in range(self.size(0)):
        res.append(tolist(self[i]))
    return res

Tensor.tolist = tolist



