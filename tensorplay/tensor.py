from . import _C
import tensorplay

# Re-export Tensor class
# In the future, we can extend this class or wrap it to add pure Python functionality
class Tensor(_C.TensorBase):
    __name__ = "Tensor"
    __module__ = "tensorplay"

    def is_float(self) -> bool:
        """
        Check if tensor is floating point.
        """
        return self.dtype == _C.DType.float32 or self.dtype == _C.DType.float64
