#include "tensorplay/core/Tensor.h"
#include "tensorplay/core/Dispatcher.h"
#include "tensorplay/core/Scalar.h"
#include "tensorplay/core/TypePromotion.h"
#include "tensorplay/utils/Utils.h"
#include "tensorplay/core/Exception.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace tensorplay {
namespace cpu {

// Helper for comparison ops
template<typename Op>
Tensor comparison_kernel_impl(const Tensor& self, const Tensor& other, Op op) {
    std::vector<int64_t> out_shape = broadcast_shapes(static_cast<std::vector<int64_t>>(self.shape()), static_cast<std::vector<int64_t>>(other.shape()));
    
    // Result is always Bool
    Tensor result = Tensor::empty(out_shape, DType::Bool, self.device());
    
    // For comparison, we usually don't promote types to a common type for the operation, 
    // but C++ requires it. PyTorch promotes to common type before comparison.
    DType common_dtype = promoteTypes(self.dtype(), other.dtype());
    
    Tensor self_casted = (self.dtype() == common_dtype) ? self : self.to(common_dtype);
    Tensor other_casted = (other.dtype() == common_dtype) ? other : other.to(common_dtype);

    Tensor self_expanded = self_casted.expand(out_shape);
    Tensor other_expanded = other_casted.expand(out_shape);
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        apply_op_recursive<bool>(result.data_ptr<bool>(), result.strides(), \
                                 self_expanded, self_expanded.strides(), \
                                 other_expanded, other_expanded.strides(), \
                                 0, 0, 0, 0, out_shape, op); \
        break; \
    }

    switch (common_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(TypeError, "comparison: unsupported dtype");
    }
    #undef OP_CASE
    
    return result;
}

Tensor eq_tensor_kernel(const Tensor& self, const Tensor& other) {
    return comparison_kernel_impl(self, other, [](auto a, auto b) { return a == b; });
}

Tensor ne_tensor_kernel(const Tensor& self, const Tensor& other) {
    return comparison_kernel_impl(self, other, [](auto a, auto b) { return a != b; });
}

Tensor lt_tensor_kernel(const Tensor& self, const Tensor& other) {
    return comparison_kernel_impl(self, other, [](auto a, auto b) { return a < b; });
}

Tensor le_tensor_kernel(const Tensor& self, const Tensor& other) {
    return comparison_kernel_impl(self, other, [](auto a, auto b) { return a <= b; });
}

Tensor gt_tensor_kernel(const Tensor& self, const Tensor& other) {
    return comparison_kernel_impl(self, other, [](auto a, auto b) { return a > b; });
}

Tensor ge_tensor_kernel(const Tensor& self, const Tensor& other) {
    return comparison_kernel_impl(self, other, [](auto a, auto b) { return a >= b; });
}

// Scalar versions
Tensor eq_scalar_kernel(const Tensor& self, Scalar other) {
    Tensor other_t = Tensor::full({}, other, self.dtype(), self.device());
    return eq_tensor_kernel(self, other_t);
}

Tensor ne_scalar_kernel(const Tensor& self, Scalar other) {
    Tensor other_t = Tensor::full({}, other, self.dtype(), self.device());
    return ne_tensor_kernel(self, other_t);
}

Tensor lt_scalar_kernel(const Tensor& self, Scalar other) {
    Tensor other_t = Tensor::full({}, other, self.dtype(), self.device());
    return lt_tensor_kernel(self, other_t);
}

Tensor le_scalar_kernel(const Tensor& self, Scalar other) {
    Tensor other_t = Tensor::full({}, other, self.dtype(), self.device());
    return le_tensor_kernel(self, other_t);
}

Tensor gt_scalar_kernel(const Tensor& self, Scalar other) {
    Tensor other_t = Tensor::full({}, other, self.dtype(), self.device());
    return gt_tensor_kernel(self, other_t);
}

Tensor ge_scalar_kernel(const Tensor& self, Scalar other) {
    Tensor other_t = Tensor::full({}, other, self.dtype(), self.device());
    return ge_tensor_kernel(self, other_t);
}

TENSORPLAY_REGISTER_KERNEL(eq, CPU, eq_tensor_kernel)
TENSORPLAY_REGISTER_KERNEL_STR("eq.Scalar", CPU, eq_scalar_kernel)
TENSORPLAY_REGISTER_KERNEL(ne, CPU, ne_tensor_kernel)
TENSORPLAY_REGISTER_KERNEL_STR("ne.Scalar", CPU, ne_scalar_kernel)
TENSORPLAY_REGISTER_KERNEL(lt, CPU, lt_tensor_kernel)
TENSORPLAY_REGISTER_KERNEL_STR("lt.Scalar", CPU, lt_scalar_kernel)
TENSORPLAY_REGISTER_KERNEL(le, CPU, le_tensor_kernel)
TENSORPLAY_REGISTER_KERNEL_STR("le.Scalar", CPU, le_scalar_kernel)
TENSORPLAY_REGISTER_KERNEL(gt, CPU, gt_tensor_kernel)
TENSORPLAY_REGISTER_KERNEL_STR("gt.Scalar", CPU, gt_scalar_kernel)
TENSORPLAY_REGISTER_KERNEL(ge, CPU, ge_tensor_kernel)
TENSORPLAY_REGISTER_KERNEL_STR("ge.Scalar", CPU, ge_scalar_kernel)

} // namespace cpu
} // namespace tensorplay
